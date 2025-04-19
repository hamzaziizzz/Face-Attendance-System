import cv2
import numpy as np
import onnxruntime as ort
import threading
import queue
import time
from pymilvus import connections, Collection
from face_detection import FaceDetectorSCRFD

# --- Global Initialization ---
# 1) GPU-based SCRFD detector
detector = FaceDetectorSCRFD(
    model_path="models/detection/onnx/scrfd_10g_gnkps.onnx",
    input_size=(640, 640),
    conf_thresh=0.5,
    nms_thresh=0.4
)
# 2) ArcFace recognition session
rec_session = ort.InferenceSession(
    "models/recognition/onnx/w600k_r50.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
rec_input_name = rec_session.get_inputs()[0].name
# 3) Milvus connection & collection
connections.connect(alias="default", host="192.168.12.1", port="19530")
collection = Collection("ABESIT_FACE_DATA_COLLECTION_FOR_COSINE")
collection.load()

# --- Helper Functions ---
def preprocess_recognition(face_img: np.ndarray) -> np.ndarray:
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (112, 112), interpolation=cv2.INTER_AREA)
    face_norm = face_resized.astype(np.float32) / 255.0
    face_norm = (face_norm - 0.5) / 0.5
    return face_norm.transpose(2, 0, 1)[None, ...]

# --- Worker Function ---
def worker(rtsp_url, out_queue):
    cap = cv2.VideoCapture(rtsp_url)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        orig_h, orig_w = frame.shape[:2]
        det_frame = cv2.resize(frame, (640, 360))
        for x1, y1, x2, y2, score in detector.detect(det_frame):
            sx, sy = orig_w / 640, orig_h / 360
            ox1, oy1 = int(x1 * sx), int(y1 * sy)
            ox2, oy2 = int(x2 * sx), int(y2 * sy)
            face_crop = frame[oy1:oy2, ox1:ox2]
            if face_crop.size == 0:
                continue
            blob = preprocess_recognition(face_crop)
            emb = rec_session.run(None, {rec_input_name: blob})[0].flatten().tolist()
            params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
            res = collection.search(
                data=[emb], anns_field="embeddings", param=params,
                limit=1, output_fields=["name_id"]
            )
            color = (0, 0, 255)
            label = ""
            if res and len(res[0]) > 0:
                hit = res[0][0]
                name_id = hit.entity.get("name_id")
                match_score = hit.score
                color = (0, 255, 0) if match_score >= 0.4 else (0, 0, 255)
                label = f"{name_id}: {match_score:.2f}"
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), color, 2)
            if label:
                cv2.putText(frame, label, (ox1, oy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        out_queue.put(frame)
    cap.release()

# --- Main ---
def main():
    streams = [
        "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        0,
    ]
    queues = {}
    threads = []
    for url in streams:
        q = queue.Queue(maxsize=1)
        queues[url] = q
        t = threading.Thread(target=worker, args=(url, q), daemon=True)
        t.start()
        threads.append(t)

    # Main display loop
    try:
        while True:
            for url, q in queues.items():
                if not q.empty():
                    frame = q.get()
                    window = f"Stream: {url}"
                    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                    # cv2.setWindowProperty(
                    #     window, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                    # )
                    cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
