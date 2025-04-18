import onnxruntime as ort
import numpy as np
import math
import cv2

class FaceDetectorSCRFD:
    def __init__(self,
                 model_path: str,
                 input_size=(640, 640),
                 conf_thresh=0.5,   # default threshold for face confidence
                 nms_thresh=0.4):
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        # Use CUDA if available for speed
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _nms(self, boxes, scores, iou_thresh):
        idxs = np.argsort(scores)[::-1]
        keep = []
        while len(idxs) > 0:
            curr = idxs[0]
            keep.append(curr)
            if len(idxs) == 1:
                break
            ious = self._iou(boxes[curr], boxes[idxs[1:]])
            idxs = idxs[1:][ious < iou_thresh]
        return keep

    def _iou(self, box1, boxes):
        x1 = np.maximum(box1[0], boxes[:, 0])
        y1 = np.maximum(box1[1], boxes[:, 1])
        x2 = np.minimum(box1[2], boxes[:, 2])
        y2 = np.minimum(box1[3], boxes[:, 3])
        inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - inter
        return inter / (union + 1e-6)

    def preprocess(self, image):
        h0, w0 = image.shape[:2]
        target_h, target_w = self.input_size
        # letterbox resize
        scale = min(target_w / w0, target_h / h0)
        nh, nw = int(h0 * scale), int(w0 * scale)
        resized = cv2.resize(image, (nw, nh))
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        top = (target_h - nh) // 2
        left = (target_w - nw) // 2
        padded[top:top+nh, left:left+nw] = resized
        # BGR to RGB
        rgb = padded[..., ::-1]
        # normalize to [-1,1]
        img = rgb.astype(np.float32)
        img = (img - 127.5) / 128.0
        # CHW + batch
        img = img.transpose(2, 0, 1)[None, ...]
        return img, (h0, w0), (scale, top, left)

    def postprocess(self, outputs, original_shape, scale_info):
        scale, top, left = scale_info
        h0, w0 = original_shape
        strides = [8, 16, 32]
        all_boxes = []
        all_scores = []
        # loop over strides
        for i, stride in enumerate(strides):
            # confidence scores
            scores = outputs[i].reshape(-1)
            # bbox deltas: output index offset by len(strides)
            d = outputs[i + len(strides)].squeeze(0)
            # number of anchors per location = 2 for scrfd_10g
            num_anchors = 2
            # grid size
            N = d.shape[0]
            cells = int(math.sqrt(N / num_anchors))
            # build grid of centers
            ys, xs = np.meshgrid(np.arange(cells), np.arange(cells), indexing='ij')
            centers = np.stack((xs, ys), axis=-1).reshape(-1, 2) * stride
            centers = np.repeat(centers, num_anchors, axis=0)
            # decode distances
            l = d[:, 0] * stride
            t = d[:, 1] * stride
            r = d[:, 2] * stride
            b = d[:, 3] * stride
            x1 = centers[:, 0] - l
            y1 = centers[:, 1] - t
            x2 = centers[:, 0] + r
            y2 = centers[:, 1] + b
            # apply confidence threshold
            mask = scores > self.conf_thresh
            if mask.sum() == 0:
                continue
            boxes = np.stack([x1, y1, x2, y2], axis=1)[mask]
            scs = scores[mask]
            all_boxes.append(boxes)
            all_scores.append(scs)
        if not all_boxes:
            return []
        # concatenate detections
        boxes = np.vstack(all_boxes)
        scores = np.hstack(all_scores)
        # NMS
        keep = self._nms(boxes, scores, self.nms_thresh)
        boxes = boxes[keep]
        scores = scores[keep]
        # scale back to original
        results = []
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            x1 = int(max(0, min(w0, (x1 - left) / scale)))
            y1 = int(max(0, min(h0, (y1 - top) / scale)))
            x2 = int(max(0, min(w0, (x2 - left) / scale)))
            y2 = int(max(0, min(h0, (y2 - top) / scale)))
            results.append((x1, y1, x2, y2, float(score)))
        return results

    def detect(self, image):
        img_input, original_shape, scale_info = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: img_input})
        return self.postprocess(outputs, original_shape, scale_info)

if __name__ == "__main__":
    detector = FaceDetectorSCRFD(
        model_path="models/detection/onnx/scrfd_10g_gnkps.onnx",
        conf_thresh=0.6,
        nms_thresh=0.4
    )
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("/home/hamza/OfficeProjects/CCTV-Face-Recognition-Attendance-System/TEST-VIDEOS/ABESIT/ABESIT-Main-Gate-Test-Video.avi")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        detections = detector.detect(frame)
        for x1, y1, x2, y2, score in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{score:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 2)
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()