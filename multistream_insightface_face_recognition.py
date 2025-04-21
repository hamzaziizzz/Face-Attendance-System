import math
import time
import queue
import threading

import cv2
import numpy as np
from pymilvus import connections, Collection
from insightface_rest_api import IFRClient


ifr_client = IFRClient()

connections.connect(alias="default", host="localhost", port="19530")
collection = Collection("ABESIT_FACE_DATA_COLLECTION_FOR_COSINE")
collection.load()

params = {"metric_type": "COSINE", "params": {"nprobe": 16}}


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

        # orig_h, orig_w = frame.shape[:2]
        # det_frame = cv2.resize(frame, (640, 360))

        face_data = ifr_client.extract_face_data(frame)

        # print()
        # print(face_data["data"][0]["faces"])
        # print()

        for face in face_data["data"][0]["faces"]:
            x_min, y_min, x_max, y_max = face["bbox"]
            encoding = np.array(face["vec"])

            res = collection.search(
                data=[encoding], anns_field="embeddings", param=params,
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

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            if label:
                cv2.putText(frame, label, (x_min, y_min - 10),
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
    cv2.namedWindow("InsightFace-REST-API Sequential Pipeline", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("InsightFace-REST-API Sequential Pipeline",
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

            cv2.imshow("InsightFace-REST-API Sequential Pipeline", mosaic)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.01)
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
