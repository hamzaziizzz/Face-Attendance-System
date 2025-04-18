import cv2
import numpy as np
import onnxruntime as ort
from pymilvus import connections, Collection
from face_detection import FaceDetectorSCRFD

# --- Initialize SCRFD face detector (low-res) ---
detector = FaceDetectorSCRFD(
    model_path="models/detection/onnx/scrfd_10g_gnkps.onnx",
    input_size=(640, 640),
    conf_thresh=0.5,
    nms_thresh=0.4
)

# --- Initialize ArcFace recognition session ---
rec_session = ort.InferenceSession(
    "models/recognition/onnx/w600k_r50.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
rec_input_name = rec_session.get_inputs()[0].name

# --- Connect to Milvus ---
connections.connect(
    alias="default",
    host="192.168.12.1",      # update as needed
    port="19530"
)
collection = Collection("ABESIT_FACE_DATA_COLLECTION_FOR_COSINE")
collection.load()

# --- Recognition preprocessing ---
def preprocess_recognition(face_img: np.ndarray) -> np.ndarray:
    # BGR -> RGB, resize to 112x112, normalize to [-1,1], CHW + batch
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (112, 112))
    face_norm = face_resized.astype(np.float32) / 255.0
    face_norm = (face_norm - 0.5) / 0.5
    blob = face_norm.transpose(2, 0, 1)[None, ...]
    return blob

# --- Video capture and pipeline ---
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi")
if not cap.isOpened():
    raise RuntimeError("Cannot open camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]
    # Downscale for detection
    det_w, det_h = 640, 360
    det_frame = cv2.resize(frame, (det_w, det_h))

    # Detect faces (low-res)
    detections = detector.detect(det_frame)

    # Process each detected face
    for x1, y1, x2, y2, score in detections:
        # Map box coords to original frame
        scale_x = orig_w / det_w
        scale_y = orig_h / det_h
        ox1, oy1 = int(x1 * scale_x), int(y1 * scale_y)
        ox2, oy2 = int(x2 * scale_x), int(y2 * scale_y)

        # Crop high-res face
        face_crop = frame[oy1:oy2, ox1:ox2]
        if face_crop.size == 0:
            continue

        # Extract embedding
        blob = preprocess_recognition(face_crop)
        embedding = rec_session.run(None, {rec_input_name: blob})[0]
        emb_list = embedding.flatten().tolist()

        # Milvus search
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        results = collection.search(
            data=[emb_list],
            anns_field="embeddings",
            param=search_params,
            limit=1,
            output_fields=["name_id"]
        )

        # Annotate frame with match
        if results and len(results[0]) > 0:
            hit = results[0][0]
            name_id = hit.entity.get("name_id")
            match_score = hit.score
            if match_score >= 0.4:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), color, 2)
            label = f"{name_id}: {match_score:.2f}"
            cv2.putText(
                frame, label, (ox1, oy1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2
            )

            # Draw detection box
            cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), color, 2)

    cv2.imshow("Face Matching", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
