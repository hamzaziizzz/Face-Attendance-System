#!/usr/bin/env python3
"""
Decoupled batched face-detection/recognition on multiple video streams,
with smooth display. Readers update raw_frames at full rate; inference
batches N frames at a time, writes detections; UI loop draws every frame.
"""

import math
import time
import queue
import threading

import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit                 # one global primary context
from pymilvus import connections, Collection

# ─── Config ────────────────────────────────────────────────────────────────────
DETECTION_ENGINE   = "models/detection/trt-engine/scrfd_10g_gnkps_dynamic.engine"
RECOGNITION_ENGINE = "models/recognition/trt-engine/w600k_r50_dynamic.engine"
INPUT_SIZE_DET     = (640, 640)     # H, W into the TRT engine
DET_CANVAS         = (640, 360)     # width, height before letterbox
RECOG_INPUT_SIZE   = (112, 112)
CONF_TH            = 0.5
NMS_TH             = 0.4

# replace with your actual video sources or RTSP URLs
STREAMS     = [
    0,
    # "rtsp://grilsquad:grilsquad@192.168.12.18:554/stream1",
    # "rtsp://grilsquad:grilsquad@192.168.12.17:554/stream1",
    # "rtsp://grilsquad:grilsquad@192.168.12.19:554/stream1",
    # "rtsp://admin:admin@123@192.168.7.61:554/cam/realmonitor?channel=1&subtype=0",
    # "rtsp://admin:admin@123@192.168.7.60:554/cam/realmonitor?channel=1&subtype=0",
]

MOSAIC_CELL = (360, 640)  # H, W of each thumbnail
BATCH_SIZE  = 4
BATCH_TIMEOUT = 0.03      # seconds

# ─── TRT / CUDA Helpers ────────────────────────────────────────────────────────
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def load_engine(path: str) -> trt.ICudaEngine:
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(ctx, engine, actual_batch=1):
    inputs, outputs, bindings = [], [], []
    for i in range(engine.num_io_tensors):
        name  = engine.get_tensor_name(i)
        shape = list(ctx.get_tensor_shape(name))

        # Replace batch dim (0th index) with actual batch size
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            shape[0] = actual_batch
        else:
            if shape[0] == -1:  # for dynamic outputs
                shape[0] = actual_batch

        dtype = trt.nptype(engine.get_tensor_dtype(name))
        size  = int(np.prod(shape))
        h_mem = cuda.pagelocked_empty(size, dtype)
        d_mem = cuda.mem_alloc(h_mem.nbytes)
        ctx.set_tensor_address(name, int(d_mem))
        bindings.append(int(d_mem))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            inputs.append((h_mem, d_mem))
        else:
            outputs.append((h_mem, d_mem))
    return inputs, outputs, bindings


# ─── Batched Face Detector ─────────────────────────────────────────────────────
class FaceDetectorTRT:
    def __init__(self, engine_path, input_size, conf_th, nms_th):
        self.H, self.W = input_size
        self.conf_th, self.nms_th = conf_th, nms_th
        self.strides = [8, 16, 32]

        self.engine = load_engine(engine_path)
        self.ctx = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.input_name = next(
            name for name in [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        )

    def _letterbox(self, img):
        h0, w0 = img.shape[:2]
        scale = min(self.W / w0, self.H / h0)
        nh, nw = int(h0 * scale), int(w0 * scale)
        resized = cv2.resize(img, (nw, nh))
        pad = np.full((self.H, self.W, 3), 114, dtype=np.uint8)
        top, left = (self.H - nh) // 2, (self.W - nw) // 2
        pad[top:top + nh, left:left + nw] = resized
        rgb = pad[..., ::-1].astype(np.float32)
        norm = (rgb - 127.5) / 128.0
        blob = norm.transpose(2, 0, 1)
        return blob, (h0, w0), (scale, top, left)

    def preprocess_batch(self, frames):
        B = len(frames)
        blobs = np.empty((B, 3, self.H, self.W), dtype=np.float32)
        metas = []
        for i, f in enumerate(frames):
            blob, orig_hw, affine = self._letterbox(f)
            blobs[i] = blob
            metas.append((orig_hw, affine))
        return blobs, metas

    def infer(self, blobs):
        B = blobs.shape[0]
        self.ctx.set_input_shape(self.input_name, (B, 3, self.H, self.W))

        h_in = cuda.pagelocked_empty(B * 3 * self.H * self.W, dtype=np.float32)
        h_in[:] = 0.0
        np.copyto(h_in, blobs.ravel())
        d_in = cuda.mem_alloc(h_in.nbytes)
        cuda.memcpy_htod_async(d_in, h_in, self.stream)
        self.ctx.set_tensor_address(self.input_name, int(d_in))

        outputs = []
        out_mems = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = self.ctx.get_tensor_shape(name)
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                size = int(np.prod(shape))
                h_out = cuda.pagelocked_empty(size, dtype)
                d_out = cuda.mem_alloc(h_out.nbytes)
                self.ctx.set_tensor_address(name, int(d_out))
                out_mems.append((h_out, d_out))
                outputs.append((name, h_out))

        self.ctx.execute_async_v3(self.stream.handle)
        for h_out, d_out in out_mems:
            cuda.memcpy_dtoh_async(h_out, d_out, self.stream)
        self.stream.synchronize()

        return outputs, B

    def postprocess_batch(self, outputs, metas):
        B = len(metas)
        output_map = {name: out.reshape(B, -1, out.shape[-1] if out.ndim == 3 else 1) for name, out in outputs}
        scores = [output_map[f"score_{s}"] for s in self.strides]
        bboxes = [output_map[f"bbox_{s}"] for s in self.strides]

        detections = [[] for _ in range(B)]
        for idx in range(B):
            boxes, confs = [], []
            for i, s in enumerate(self.strides):
                sc = scores[i][idx].reshape(-1)
                bb = bboxes[i][idx].reshape(-1, 4)
                N = bb.shape[0]
                side = int(math.sqrt(N // 2))
                ys, xs = np.meshgrid(range(side), range(side), indexing="ij")
                ctr = np.stack((xs, ys), -1).reshape(-1, 2) * s
                ctr = np.repeat(ctr, 2, axis=0)
                l, t, r, b = [bb[:, j] * s for j in range(4)]
                x1 = ctr[:, 0] - l; y1 = ctr[:, 1] - t
                x2 = ctr[:, 0] + r; y2 = ctr[:, 1] + b
                mask = sc > self.conf_th
                if mask.any():
                    boxes.append(np.stack([x1, y1, x2, y2], 1)[mask])
                    confs.append(sc[mask])
            if not boxes: continue
            boxes = np.vstack(boxes)
            confs = np.hstack(confs)
            keep = []
            idxs = np.argsort(confs)[::-1]
            while idxs.size:
                i = idxs[0]; keep.append(i)
                if idxs.size == 1: break
                ious = self._iou(boxes[i], boxes[idxs[1:]])
                idxs = idxs[1:][ious < self.nms_th]
            (h0, w0), (scale, top, left) = metas[idx]
            for i in keep:
                x1, y1, x2, y2 = boxes[i]
                xi1 = int(max(0, min(w0, (x1 - left) / scale)))
                yi1 = int(max(0, min(h0, (y1 - top ) / scale)))
                xi2 = int(max(0, min(w0, (x2 - left) / scale)))
                yi2 = int(max(0, min(h0, (y2 - top ) / scale)))
                detections[idx].append((xi1, yi1, xi2, yi2, float(confs[i])))
        return detections

    @staticmethod
    def _iou(b1, boxes):
        x1 = np.maximum(b1[0], boxes[:, 0])
        y1 = np.maximum(b1[1], boxes[:, 1])
        x2 = np.minimum(b1[2], boxes[:, 2])
        y2 = np.minimum(b1[3], boxes[:, 3])
        inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
        a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return inter / (a1 + a2 - inter + 1e-6)


# ─── Per-face Recognizer ───────────────────────────────────────────────────────
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


# ─── Reader ────────────────────────────────────────────────────────────────────
def reader(rtsp_url, frame_queue, raw_frames):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"[reader] ERROR opening {rtsp_url}")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            # time.sleep(0.01)
            continue
        # if isinstance(rtsp_url, int):
        #     print(f"[reader-debug] VideoFrame Shape: {frame.shape}, dtype: {frame.dtype}, mean: {np.mean(frame):.2f}")
        raw_frames[rtsp_url] = frame
        if frame_queue.full():
            try: frame_queue.get_nowait()
            except queue.Empty: pass
        frame_queue.put((rtsp_url, frame))
    cap.release()

# ─── Inference ─────────────────────────────────────────────────────────────────
def inferencer(frame_queue, raw_frames, detections,
               det_engine, rec_engine, coll, search_param):
    cuda_ctx = cuda.Device(0).make_context()
    try:
        detector   = FaceDetectorTRT(det_engine,
                                     INPUT_SIZE_DET, CONF_TH, NMS_TH)
        recognizer = FaceRecognizerTRT(rec_engine)

        while True:
            batch, t0 = [], time.time()
            while len(batch) < BATCH_SIZE:
                try:
                    url, frame = frame_queue.get(
                        timeout=max(0, BATCH_TIMEOUT-(time.time()-t0))
                    )
                    batch.append((url, frame))
                except queue.Empty:
                    break
            if not batch:
                continue

            urls, frames = zip(*batch)
            det_frames = [
                cv2.resize(f, DET_CANVAS, interpolation=cv2.INTER_AREA)
                for f in frames
            ]

            blobs, metas = detector.preprocess_batch(det_frames)
            outs, actual_B = detector.infer(blobs)
            dets_b       = detector.postprocess_batch(outs, metas)

            # update only detections[url] with scaled boxes + labels
            for idx, (url, frame) in enumerate(batch):
                dets = dets_b[idx]
                # if dets: print(dets)
                h, w = frame.shape[:2]
                sx, sy = w/DET_CANVAS[0], h/DET_CANVAS[1]
                annots = []
                for (x1,y1,x2,y2,score) in dets:
                    ox1 = int(round(x1*sx)); oy1 = int(round(y1*sy))
                    ox2 = int(round(x2*sx)); oy2 = int(round(y2*sy))
                    # cv2.rectangle(frame,(ox1,oy1),(ox2,oy2),(0,0,255),2)
                    crop = frame[oy1:oy2, ox1:ox2]
                    if crop.size==0: continue
                    blob = preprocess_recognition(crop)
                    # print("=================================================================")
                    # print(np.min(crop), np.max(crop))  # should be 0-255
                    # print(np.min(blob), np.max(blob))  # should be ~ -1.0 to 1.0
                    # print("=================================================================")
                    emb = recognizer.infer(blob)[0]
                    # print(emb.tolist())
                    res = coll.search([emb.tolist()],
                                      anns_field="embeddings",
                                      param=search_param,
                                      limit=1,
                                      output_fields=["name_id"])
                    # print(res)
                    label, color = "", (0,0,255)
                    if res and len(res[0])>0:
                        hit = res[0][0]
                        sc, nm = hit.score, hit.entity.get("name_id")
                        color = (0,255,0) if sc>=0.4 else (0,0,255)
                        label = f"{nm}:{sc:.2f}"
                    annots.append((ox1,oy1,ox2,oy2,label,color))
                detections[url] = annots

    finally:
        cuda_ctx.pop()
        cuda_ctx.detach()

# ─── Main & UI ─────────────────────────────────────────────────────────────────
def main():
    # Milvus connect
    connections.connect(alias="default", host="localhost", port="19530")
    coll = Collection("ABESIT_FACE_DATA_COLLECTION_FOR_COSINE")
    coll.load()
    search_param = {"metric_type":"COSINE","params":{"nprobe":16}}

    frame_queue = queue.Queue(maxsize=BATCH_SIZE*2)
    raw_frames  = {u: np.zeros((MOSAIC_CELL[0],MOSAIC_CELL[1],3),dtype=np.uint8) for u in STREAMS}
    detections  = {u: [] for u in STREAMS}

    # start readers
    for url in STREAMS:
        threading.Thread(target=reader,
                         args=(url, frame_queue, raw_frames),
                         daemon=True).start()

    # start inference
    threading.Thread(target=inferencer,
                     args=(frame_queue, raw_frames, detections,
                           DETECTION_ENGINE, RECOGNITION_ENGINE,
                           coll, search_param),
                     daemon=True).start()

    # UI-mosaic
    Hc, Wc = MOSAIC_CELL
    cols = math.ceil(math.sqrt(len(STREAMS)))
    rows = math.ceil(len(STREAMS)/cols)
    cv2.namedWindow("FaceRec TRT Mosaic", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("FaceRec TRT Mosaic", Wc*cols, Hc*rows)

    # VIDEO SAVING ─────────────────────────────────────────────
    out_path = "output_inferred_video.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Or 'MJPG'
    out_fps = 25  # Set to actual frame rate if known
    out_size = (Wc * cols, Hc * rows)
    video_writer = cv2.VideoWriter(out_path, fourcc, out_fps, out_size)

    while True:
        thumbs = []
        for url in STREAMS:
            frm = raw_frames[url].copy()
            for (x1,y1,x2,y2,label,color) in detections[url]:
                cv2.rectangle(frm, (x1,y1), (x2,y2), color, 2)
                if label:
                    cv2.putText(frm, label, (x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            thumbs.append(cv2.resize(frm, (Wc,Hc), interpolation=cv2.INTER_AREA))

        mosaic = np.zeros((Hc*rows, Wc*cols, 3), dtype=np.uint8)
        for i,thumb in enumerate(thumbs):
            r,c = divmod(i, cols)
            mosaic[r*Hc:(r+1)*Hc, c*Wc:(c+1)*Wc] = thumb

        # SAVE TO VIDEO
        video_writer.write(mosaic)

        cv2.imshow("FaceRec TRT Mosaic", mosaic)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
