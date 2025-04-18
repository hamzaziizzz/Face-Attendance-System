import cv2
import numpy as np
import onnxruntime as ort
from face_detection import FaceDetectorSCRFD

# --- Initialize face detector on low-res stream ---
detector = FaceDetectorSCRFD(
    model_path="models/detection/onnx/scrfd_10g_gnkps.onnx",
    input_size=(640, 640),
    conf_thresh=0.5,
    nms_thresh=0.4
)

# --- Initialize face recognition model ---
rec_session = ort.InferenceSession(
    "models/recognition/onnx/w600k_r50.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
rec_input_name = rec_session.get_inputs()[0].name

# --- Preprocessing function for recognition model ---
def preprocess_recognition(face_img: np.ndarray) -> np.ndarray:
    """
    Convert BGR face crop to RGB, resize to 112x112, normalize to [-1,1], and add batch dimension.
    """
    # BGR -> RGB
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    # Resize to 112x112
    face_resized = cv2.resize(face_rgb, (112, 112), interpolation=cv2.INTER_AREA)
    # Normalize to [0,1]
    face_norm = face_resized.astype(np.float32) / 255.0
    # Standardize to [-1,1]
    face_norm = (face_norm - 0.5) / 0.5
    # CHW and batch
    blob = face_norm.transpose(2, 0, 1)[None, ...]
    return blob

# --- Video capture and pipeline ---
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi")
if not cap.isOpened():
    raise RuntimeError("Could not open video capture")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    orig_h, orig_w = frame.shape[:2]
    # Downscale for detection: 640x360 (aspect ratio preserved for 16:9)
    det_w, det_h = 640, 360
    det_frame = cv2.resize(frame, (det_w, det_h))

    # Detect faces on the low-res frame
    detections = detector.detect(det_frame)

    # For each detected face, crop from original and recognize
    for x1, y1, x2, y2, score in detections:
        # Map det-frame coords back to original resolution
        scale_x = orig_w / det_w
        scale_y = orig_h / det_h
        ox1 = int(x1 * scale_x)
        oy1 = int(y1 * scale_y)
        ox2 = int(x2 * scale_x)
        oy2 = int(y2 * scale_y)
        # Crop high-res face
        face_crop = frame[oy1:oy2, ox1:ox2]
        if face_crop.size == 0:
            continue
        # Prepare for recognition
        recog_blob = preprocess_recognition(face_crop)
        # Extract features
        embedding = rec_session.run(None, {rec_input_name: recog_blob})[0]
        print(embedding)

        # Draw detection box and score
        cv2.rectangle(frame, (ox1, oy1), (ox2, oy2), (0, 255, 0), 2)
        cv2.putText(
            frame, f"{score:.2f}", (ox1, oy1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2
        )

    # Show the resulting frame
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
