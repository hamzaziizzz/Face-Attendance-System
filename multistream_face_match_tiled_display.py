import math
import time
import queue
import threading

import cv2
import numpy as np
import onnxruntime as ort
from pymilvus import connections, Collection

from face_detection import FaceDetectorSCRFD

# --- Global Initialization ---
# 1) GPU‑based SCRFD detector
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
connections.connect(alias="default", host="localhost", port="19530")
collection = Collection("ABESIT_FACE_DATA_COLLECTION_FOR_COSINE")
collection.load()

params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

def preprocess_recognition(face_img: np.ndarray) -> np.ndarray:
    """Convert BGR→RGB, resize to 112×112, normalize, and add batch dim."""
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (112, 112), interpolation=cv2.INTER_AREA)
    face_norm = face_resized.astype(np.float32) / 255.0
    face_norm = (face_norm - 0.5) / 0.5
    return face_norm.transpose(2, 0, 1)[None, ...]


def worker(rtsp_url, out_queue):
    """Read frames, detect & recognize faces, draw overlays, and push to queue."""
    cap = cv2.VideoCapture(rtsp_url)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # Counters and timers
    # in_count = 0
    out_count = 0
    in_t0 = time.time()
    out_t0 = in_t0
    # in_fps = 0.0
    out_fps = 0.0

    while cap.isOpened():
        t_read = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # # —— INPUT FPS ——
        # in_count += 1
        # dt_in = t_read - in_t0
        # if dt_in >= 1.0:
        #     in_fps = in_count / dt_in
        #     in_count = 0
        #     in_t0 = t_read
        # # Draw input‐FPS in top‐left
        # cv2.putText(frame, f"In: {in_fps:.2f}FPS",
        #             (30, 80),
        #             cv2.FONT_HERSHEY_SIMPLEX, 3,
        #             (0,255,255), 8)

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

            res = collection.search(
                data=[emb], anns_field="embeddings", param=params,
                limit=1, output_fields=["name_id"]
            )

            color = (0, 0, 255)
            label = ""
            if res and len(res[0]) > 0:
                hit = res[0][0]
                match_score = hit.score
                name_id = hit.entity.get("name_id")
                color = (0, 255, 0) if match_score >= 0.4 else (0, 0, 255)
                label = f"{name_id}: {match_score:.2f}"

            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), color, 2)
            if label:
                cv2.putText(frame, label, (ox1, oy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

        # —— OUTPUT FPS ——
        t_proc = time.time()
        out_count += 1
        dt_out = t_proc - out_t0
        if dt_out >= 1.0:
            out_fps = out_count / dt_out
            out_count = 0
            out_t0 = t_proc

        # Draw output‐FPS in bottom‐left
        txt = f"FPS: {out_fps:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, txt, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 8)


        # drop old frame if queue full, then push current
        if out_queue.full():
            try:
                out_queue.get_nowait()
            except queue.Empty:
                pass
        out_queue.put(frame)

    cap.release()


def main():
    # list your sources here (file paths, RTSP URLs, or camera indices)
    streams = [
        "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
        "/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi",
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
        # 0,
        # add more streams if needed
    ]

    # spawn one thread + queue per stream
    queues = {}
    for url in streams:
        q = queue.Queue(maxsize=1)
        queues[url] = q
        t = threading.Thread(target=worker, args=(url, q), daemon=True)
        t.start()

    # thumbnail size (adjust as needed)
    THUMB_W, THUMB_H = 640, 360

    # compute a near‑square grid
    n = len(streams)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    cv2.namedWindow("Detection-Recognition Parallel Pipeline", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Detection-Recognition Parallel Pipeline",
                          cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_NORMAL)

    # cache last thumbnail for each stream
    last_thumbs = {
        url: np.zeros((THUMB_H, THUMB_W, 3), dtype=np.uint8)
        for url in streams
    }

    try:
        while True:
            thumbs = []
            for url in streams:
                q = queues[url]
                if not q.empty():
                    frame = q.get()
                    thumb = cv2.resize(frame, (THUMB_W, THUMB_H),
                                       interpolation=cv2.INTER_AREA)
                    last_thumbs[url] = thumb
                thumbs.append(last_thumbs[url])

            # build mosaic canvas
            mosaic = np.zeros((THUMB_H * rows, THUMB_W * cols, 3),
                              dtype=np.uint8)
            for idx, thumb in enumerate(thumbs):
                r, c = divmod(idx, cols)
                y0, x0 = r * THUMB_H, c * THUMB_W
                mosaic[y0:y0 + THUMB_H, x0:x0 + THUMB_W] = thumb

            cv2.imshow("Detection-Recognition Parallel Pipeline", mosaic)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
