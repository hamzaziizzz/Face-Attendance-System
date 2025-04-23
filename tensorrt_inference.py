import math
import time
import queue
import threading

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
# initialize CUDA driver (no autoinit)
cuda.init()
from pymilvus import connections, Collection

# ─── TRT HELPERS ───────────────────────────────────────────────────────────────
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(path: str) -> trt.ICudaEngine:
    with open(path, "rb") as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(context, engine: trt.ICudaEngine):
    inputs, outputs, bindings = [], [], []
    for idx in range(engine.num_io_tensors):
        name  = engine.get_tensor_name(idx)
        shape = context.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size  = int(np.prod(shape))
        host_mem = cuda.pagelocked_empty(size, dtype)
        dev_mem  = cuda.mem_alloc(host_mem.nbytes)
        context.set_tensor_address(name, int(dev_mem))
        bindings.append(int(dev_mem))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append((host_mem, dev_mem))
        else:
            outputs.append((host_mem, dev_mem))
    return inputs, outputs, bindings

# ─── TRT FACE DETECTOR ─────────────────────────────────────────────────────────
class FaceDetectorTRT:
    def __init__(self, engine_path: str,
                 input_size=(640,640), conf_thresh=0.5, nms_thresh=0.4):
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh  = nms_thresh
        self.strides     = [8,16,32]

        self.engine  = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        # bind input shape
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, (1,3,*self.input_size))
                break
        self.inputs, self.outputs, self.bindings = allocate_buffers(self.context, self.engine)

    def preprocess(self, image: np.ndarray):
        h0,w0 = image.shape[:2]
        th,tw = self.input_size
        scale = min(tw/w0, th/h0)
        nh,nw = int(h0*scale), int(w0*scale)
        resized = cv2.resize(image, (nw,nh))
        pad = np.full((th,tw,3), 114, dtype=np.uint8)
        top,left = (th-nh)//2, (tw-nw)//2
        pad[top:top+nh, left:left+nw] = resized
        rgb = pad[...,::-1].astype(np.float32)
        norm = (rgb - 127.5)/128.0
        blob = norm.transpose(2,0,1)[None,...].ravel()
        return blob, (h0,w0), (scale, top, left)

    def _iou(self, b1, boxes):
        x1 = np.maximum(b1[0], boxes[:,0]); y1 = np.maximum(b1[1], boxes[:,1])
        x2 = np.minimum(b1[2], boxes[:,2]); y2 = np.minimum(b1[3], boxes[:,3])
        inter = np.maximum(x2-x1,0)*np.maximum(y2-y1,0)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        return inter/(a1+a2-inter+1e-6)

    def postprocess(self, outs, original, scale_info):
        h0,w0 = original
        scale, top, left = scale_info
        scores, boxes = [], []
        Nstr = len(self.strides)
        for i, s in enumerate(self.strides):
            sc = outs[i].reshape(-1)
            d  = outs[i+Nstr].reshape(-1,4)
            num_anchors = 2
            N = d.shape[0]//num_anchors
            side = int(math.sqrt(N))
            ys,xs = np.meshgrid(range(side), range(side), indexing='ij')
            ctr = np.stack((xs,ys),-1).reshape(-1,2)*s
            ctr = np.repeat(ctr, num_anchors, axis=0)
            l,t,r,b = [d[:,j]*s for j in range(4)]
            x1 = ctr[:,0]-l; y1=ctr[:,1]-t
            x2 = ctr[:,0]+r; y2=ctr[:,1]+b
            mask = sc>self.conf_thresh
            if mask.any():
                boxes.append(np.stack([x1,y1,x2,y2],1)[mask])
                scores.append(sc[mask])
        if not boxes:
            return []
        boxes = np.vstack(boxes); scores = np.hstack(scores)
        idxs = np.argsort(scores)[::-1]
        keep = []
        while idxs.size:
            i = idxs[0]; keep.append(i)
            if idxs.size==1: break
            ious = self._iou(boxes[i], boxes[idxs[1:]])
            idxs = idxs[1:][ious<self.nms_thresh]
        results = []
        for i in keep:
            x1,y1,x2,y2 = boxes[i]
            xi1 = int(max(0,min(w0,(x1-left)/scale)))
            yi1 = int(max(0,min(h0,(y1-top)/scale)))
            xi2 = int(max(0,min(w0,(x2-left)/scale)))
            yi2 = int(max(0,min(h0,(y2-top)/scale)))
            results.append((xi1,yi1,xi2,yi2,float(scores[i])))
        return results

    def infer(self, blob):
        h_in,d_in = self.inputs[0]
        np.copyto(h_in, blob)
        cuda.memcpy_htod(d_in, h_in)
        ok = self.context.execute_v2(self.bindings)
        if not ok:
            raise RuntimeError("TensorRT execute_v2 failed")
        outs = []
        for h_out,d_out in self.outputs:
            cuda.memcpy_dtoh(h_out, d_out)
            outs.append(h_out.copy())
        return outs

# ─── TRT FACE RECOGNIZER ────────────────────────────────────────────────────────
class FaceRecognizerTRT:
    def __init__(self, engine_path: str):
        self.engine  = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            if self.engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name,(1,3,112,112))
                break
        self.inputs, self.outputs, self.bindings = allocate_buffers(self.context, self.engine)

    def infer(self, blob: np.ndarray) -> np.ndarray:
        h_in,d_in = self.inputs[0]
        np.copyto(h_in, blob.ravel())
        cuda.memcpy_htod(d_in, h_in)
        ok = self.context.execute_v2(self.bindings)
        if not ok:
            raise RuntimeError("TensorRT exec_v2 failed")
        h_out,d_out = self.outputs[0]
        cuda.memcpy_dtoh(h_out, d_out)
        return h_out.reshape(1,-1)

# ─── RECOGNITION PREPROCESS ────────────────────────────────────────────────────
def preprocess_recognition(face_img: np.ndarray) -> np.ndarray:
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (112, 112), interpolation=cv2.INTER_AREA)
    norm = face_resized.astype(np.float32) / 255.0
    norm = (norm - 0.5) / 0.5
    return norm.transpose(2, 0, 1)[None, ...]


# ─── WORKER & DISPLAY ─────────────────────────────────────────────────────────
def worker(rtsp_url, out_queue, det_engine, rec_engine, collection, params):
    # create CUDA context for this thread
    ctx = cuda.Device(0).make_context()
    try:
        detector   = FaceDetectorTRT(det_engine)
        recognizer = FaceRecognizerTRT(rec_engine)
        cap = cv2.VideoCapture(rtsp_url)
        out_count, out_t0, out_fps = 0, time.time(), 0.0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            h,w = frame.shape[:2]
            det_frame = cv2.resize(frame,(640,360),interpolation=cv2.INTER_AREA)
            blob, orig, info = detector.preprocess(det_frame)
            outs = detector.infer(blob)
            dets = detector.postprocess(outs, orig, info)

            if dets: print(dets)
            for x1,y1,x2,y2,score in dets:
                sx,sy = w/640, h/360
                ox1,oy1 = int(x1*sx), int(y1*sy)
                ox2,oy2 = int(x2*sx), int(y2*sy)
                crop = frame[oy1:oy2, ox1:ox2]
                if crop.size==0: continue
                # pass preprocessed blob, not raw crop
                emb = recognizer.infer(preprocess_recognition(crop))[0].tolist()
                # print(emb)
                res = collection.search([emb], anns_field="embeddings", param=params, limit=1, output_fields=["name_id"])
                # print(res)
                color, label = (0,0,255), ""
                if res and len(res[0]) > 0:
                    hit = res[0][0]
                    match_score = hit.score
                    name_id = hit.entity.get("name_id")
                    color = (0, 255, 0) if match_score >= 0.4 else (0, 0, 255)
                    label = f"{name_id}: {match_score:.2f}"
                    # print(label)

                cv2.rectangle(frame,(ox1,oy1),(ox2,oy2),color,2)
                if label:
                    cv2.putText(frame,label,(ox1,oy1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,color,2)

            out_count += 1
            now = time.time()
            if now - out_t0 >= 1.0:
                out_fps = out_count/(now-out_t0)
                out_count, out_t0 = 0, now
            cv2.putText(frame, f"FPS: {out_fps:.2f}",(10,h-10),cv2.FONT_HERSHEY_SIMPLEX,3,(255,0,255),8)

            if out_queue.full():
                try: out_queue.get_nowait()
                except queue.Empty: pass
            out_queue.put(frame)
        cap.release()
    finally:
        ctx.pop()


def main():
    connections.connect(alias="default", host="localhost", port="19530")
    coll = Collection("ABESIT_FACE_DATA_COLLECTION_FOR_COSINE")
    coll.load()
    params = {"metric_type":"COSINE","params":{"nprobe":16}}

    streams = [
        # "rtsp://grilsquad:grilsquad@192.168.12.18:554/stream1",
        # "rtsp://grilsquad:grilsquad@192.168.12.17:554/stream1",
        # "rtsp://grilsquad:grilsquad@192.168.12.19:554/stream1",
        # "rtsp://admin:admin@123@192.168.7.61:554/cam/realmonitor?channel=1&subtype=0",
        # "rtsp://admin:admin@123@192.168.7.60:554/cam/realmonitor?channel=1&subtype=0",
        "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        # "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        # "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        # "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        # "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        # "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        # "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        # "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        # "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        # "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        # "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        # "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        # "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        # "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        # "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        # 0,
        # add more streams if needed
    ]

    queues = {}
    for url in streams:
        q = queue.Queue(maxsize=1)
        t = threading.Thread(target=worker,
                             args=(url, q,
                                   "models/detection/trt-engine/scrfd_10g_gnkps.engine",
                                   "models/recognition/trt-engine/w600k_r50.engine",
                                   coll, params),
                             daemon=True)
        t.start()
        queues[url] = q

    TH, TW = 720, 1280
    n = len(streams)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n/cols)
    cv2.namedWindow("FaceRec TRT Mosaic", cv2.WINDOW_NORMAL)

    last = {u: np.zeros((TH,TW,3),dtype=np.uint8) for u in streams}
    try:
        while True:
            thumbs = []
            for u in streams:
                if not queues[u].empty():
                    last[u] = cv2.resize(queues[u].get(),(TW,TH),interpolation=cv2.INTER_AREA)
                thumbs.append(last[u])

            mosaic = np.zeros((TH*rows, TW*cols, 3),dtype=np.uint8)
            for i,thumb in enumerate(thumbs):
                r,c = divmod(i,cols)
                mosaic[r*TH:(r+1)*TH, c*TW:(c+1)*TW] = thumb

            cv2.imshow("FaceRec TRT Mosaic", mosaic)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break
            time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
