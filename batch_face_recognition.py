#!/usr/bin/env python3
"""
batch_face_recognition.py

A self-contained script to detect faces in images using a TensorRT-accelerated SCRFD detector
and extract face embeddings using a TensorRT face recognition model.

Features:
- Robust error checks and exception handling throughout.
- Comprehensive docstrings on every function, method, and class.
- Line-by-line comments explaining purpose and return values.
- NMS, letterbox resizing, buffering, dynamic batching.
"""

import os
import time
import math
import cv2
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401 initializes CUDA driver
import tensorrt as trt
from typing import List, Tuple, Dict
import gc

# Thresholds for detection confidence and NMS IoU
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4


def letterbox_resize(
        img: np.ndarray,
        target_shape: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, float, int, int]:
    """
    Resize image to fit in target_shape with unchanged aspect ratio,
    pad remaining areas with `color`. Returns:
        - canvas: the resized+padded image (HWC, BGR)
        - scale: scaling factor applied to original image
        - pad_y: vertical padding at top
        - pad_x: horizontal padding at left
    Raises:
        ValueError: if img is not a 3-channel BGR image
    """
    if img is None or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input must be a valid BGR image")
    h, w = img.shape[:2]  # original height, width
    # compute scale factor and new dims
    scale = min(target_shape[0] / h, target_shape[1] / w)
    nh, nw = int(h * scale), int(w * scale)
    # resize with linear interpolation
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # compute padding to center the image
    pad_y = (target_shape[0] - nh) // 2
    pad_x = (target_shape[1] - nw) // 2
    # create full canvas and place resized image
    canvas = np.full((target_shape[0], target_shape[1], 3),
                     color, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas, scale, pad_y, pad_x


def non_max_suppression(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_thresh: float = NMS_THRESHOLD
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Non-Maximum Suppression on bounding boxes.
    Args:
        boxes: (N,4) array of [x1,y1,x2,y2]
        scores: (N,) array of confidences
        iou_thresh: IoU threshold to suppress overlapping boxes
    Returns:
        kept_boxes: (M,4) array of surviving boxes
        kept_scores: (M,) array of surviving scores
    """
    if boxes.size == 0:
        return boxes, scores
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        # compute IoU of the highest-score box with the rest
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        # keep only boxes with IoU below threshold
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return boxes[keep], scores[keep]


class TensorRTModel:
    """
    Wrapper around a TensorRT engine for inference.
    Handles engine loading, buffer allocation, and execution.
    """

    def __init__(self, engine_path: str):
        """
        Load and deserialize the TensorRT engine from file.
        Args:
            engine_path: path to .engine file
        Raises:
            FileNotFoundError: if engine file is missing
            RuntimeError: if engine deserialization fails
        """
        if not os.path.isfile(engine_path):
            raise FileNotFoundError(f"Engine not found: {engine_path}")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with open(engine_path, "rb") as f:
            data = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(data)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # discover input/output tensor names
        self.input_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) ==
            trt.TensorIOMode.INPUT
        ]
        self.output_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i)) ==
            trt.TensorIOMode.OUTPUT
        ]
        if not self.input_names or not self.output_names:
            raise RuntimeError("Engine I/O tensors not found")

        # buffers will be created on first inference
        self.device_buffers: Dict[str, cuda.DeviceAllocation] = {}
        self.host_buffers: Dict[str, np.ndarray] = {}
        self.buffer_shapes: Dict[str, Tuple[int, ...]] = {}

    @property
    def primary_input(self) -> str:
        """Return the first (and assumed only) input tensor name."""
        return self.input_names[0]

    def infer(self, inputs: Dict[str, np.ndarray], batch_size: int) -> Dict[str, np.ndarray]:
        """
        Run inference on a batch of inputs.
        Args:
            inputs: dict mapping tensor names to numpy arrays
            batch_size: number of samples in the batch
        Returns:
            dict mapping output tensor names to numpy arrays
        Raises:
            ValueError: if input shapes don't match engine expectations
        """
        # set dynamic shapes and allocate buffers if needed
        for name in self.input_names:
            arr = inputs.get(name)
            if arr is None:
                raise ValueError(f"Missing input tensor: {name}")
            # shape check
            self.context.set_input_shape(name, arr.shape)

        for name in self.input_names + self.output_names:
            # convert Dims to tuple for comparison
            raw_shape = self.context.get_tensor_shape(name)
            shape = tuple(raw_shape)
            vol = int(np.prod(shape))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            # (re)allocate host & device buffers if shape changed
            if name not in self.host_buffers or self.buffer_shapes[name] != shape:
                self.host_buffers[name] = cuda.pagelocked_empty(vol, dtype)
                if name in self.device_buffers:
                    self.device_buffers[name].free()
                self.device_buffers[name] = cuda.mem_alloc(
                    self.host_buffers[name].nbytes)
                self.buffer_shapes[name] = shape
            # bind device buffer to engine tensor
            self.context.set_tensor_address(
                name, int(self.device_buffers[name]))

        # copy inputs to device asynchronously
        for name in self.input_names:
            cpu_buff = self.host_buffers[name]
            np.copyto(cpu_buff, inputs[name].ravel())
            cuda.memcpy_htod_async(
                self.device_buffers[name], cpu_buff, self.stream)

        # execute inference
        self.context.execute_async_v3(self.stream.handle)

        # copy outputs back to host
        for name in self.output_names:
            cuda.memcpy_dtoh_async(
                self.host_buffers[name], self.device_buffers[name], self.stream)
        self.stream.synchronize()

        # reshape host buffers into output arrays
        outputs = {}
        for name in self.output_names:
            arr = self.host_buffers[name].reshape(
                self.context.get_tensor_shape(name))
            outputs[name] = arr
        return outputs


class FaceDetector:
    """
    Face detector that uses a TensorRTModel to run SCRFD and postprocesses
    its raw outputs into bounding boxes.
    """

    def __init__(self, model_path: str, input_size: Tuple[int, int] = (640, 640)):
        """
        Initialize detector.
        Args:
            model_path: path to SCRFD TensorRT engine
            input_size: (height, width) for model input
        """
        self.model = TensorRTModel(model_path)
        self.input_size = input_size
        self.strides = [8, 16, 32]  # SCRFD output strides

    def preprocess(
            self, images: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[Tuple[int, int, float, int, int]]]:
        """
        Letterbox-resize, BGR→RGB, normalize, and stack images into batch.
        Args:
            images: list of BGR images
        Returns:
            batch: np array of shape (B,3,H,W)
            meta: list of (orig_w, orig_h, scale, pad_x, pad_y) per image
        """
        if not images:
            raise ValueError("No images provided for detection")
        batch, meta = [], []
        for img in images:
            if img is None:
                raise ValueError("Encountered None image in batch")
            h0, w0 = img.shape[:2]
            canvas, scale, pad_y, pad_x = letterbox_resize(
                img, self.input_size)
            # convert BGR→RGB float32
            rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32)
            # normalize to [-1,1]
            norm = (rgb - 127.5) / 128.0
            # reorder to CHW
            blob = norm.transpose(2, 0, 1)
            batch.append(blob)
            meta.append((w0, h0, scale, pad_x, pad_y))
        # stack into numpy array
        return np.stack(batch), meta

    def postprocess(
            self,
            outputs: Dict[str, np.ndarray],
            meta: List[Tuple[int, int, float, int, int]]
    ) -> List[List[Tuple[int, int, int, int, float]]]:
        """
        Convert raw SCRFD outputs to final boxes with NMS.
        Args:
            outputs: raw model outputs e.g. score_8, bbox_8, etc.
            meta: metadata from preprocess
        Returns:
            List of lists of (x1,y1,x2,y2,score) per image
        """
        B = len(meta)
        # reshape outputs into (B, anchors, channels)
        output_map = {}
        for name, arr in outputs.items():
            c = arr.shape[-1] if arr.ndim == 3 else 1
            output_map[name] = arr.reshape(B, -1, c)

        detections: List[List[Tuple[int, int, int, int, float]]] = [
            [] for _ in range(B)]
        # iterate each image in batch
        for b in range(B):
            boxes_all, scores_all = [], []
            # gather per-stride anchors
            for stride in self.strides:
                sc = output_map[f"score_{stride}"][b].reshape(-1)
                bb = output_map[f"bbox_{stride}"][b].reshape(-1, 4)
                N = bb.shape[0]
                side = int(math.sqrt(N//2))  # grid size
                ys, xs = np.meshgrid(range(side), range(side), indexing="ij")
                ctr = np.stack((xs, ys), -1).reshape(-1, 2) * stride
                ctr = np.repeat(ctr, 2, axis=0)  # each cell has 2 anchors
                # decode boxes
                l = bb[:, 0] * stride
                t = bb[:, 1] * stride
                r = bb[:, 2] * stride
                b_ = bb[:, 3] * stride
                x1 = ctr[:, 0] - l
                y1 = ctr[:, 1] - t
                x2 = ctr[:, 0] + r
                y2 = ctr[:, 1] + b_
                mask = sc > CONF_THRESHOLD
                if mask.any():
                    boxes_all.append(np.stack([x1, y1, x2, y2], 1)[mask])
                    scores_all.append(sc[mask])

            if not boxes_all:
                continue
            # concat and undo letterbox
            boxes = np.vstack(boxes_all)
            scores = np.hstack(scores_all)
            w0, h0, scale, pad_x, pad_y = meta[b]
            boxes[:, 0] = np.clip((boxes[:, 0]-pad_x)/scale, 0, w0)
            boxes[:, 1] = np.clip((boxes[:, 1]-pad_y)/scale, 0, h0)
            boxes[:, 2] = np.clip((boxes[:, 2]-pad_x)/scale, 0, w0)
            boxes[:, 3] = np.clip((boxes[:, 3]-pad_y)/scale, 0, h0)
            # apply NMS
            kept_boxes, kept_scores = non_max_suppression(boxes, scores)
            # collect final detections
            for (x1, y1, x2, y2), sc in zip(kept_boxes, kept_scores):
                detections[b].append(
                    (int(x1), int(y1), int(x2), int(y2), float(sc)))
        return detections

    def detect_faces(self, images: List[np.ndarray]) -> List[List[Tuple[int, int, int, int, float]]]:
        """
        Full detection pipeline: preprocess → infer → postprocess.
        Returns:
            detections per image
        """
        batch, meta = self.preprocess(images)
        outputs = self.model.infer(
            {self.model.primary_input: batch}, batch.shape[0])
        return self.postprocess(outputs, meta)


class FaceRecognizer:
    """
    Face recognizer that uses a TensorRTModel to extract face embeddings.
    """

    def __init__(self, model_path: str):
        """
        Load recognition engine.
        """
        self.model = TensorRTModel(model_path)

    def preprocess(self, faces: List[np.ndarray]) -> np.ndarray:
        """
        Resize face crops to 112×112, normalize [0,1], stack into batch.
        Returns:
            np array (B,3,112,112)
        """
        if not faces:
            return np.zeros((0, 3, 112, 112), dtype=np.float32)
        blobs = []
        for f in faces:
            if f is None or f.size == 0:
                continue
            resized = cv2.resize(f, (112, 112)).astype(np.float32) / 255.0
            blobs.append(resized.transpose(2, 0, 1))
        return np.stack(blobs) if blobs else np.zeros((0, 3, 112, 112), dtype=np.float32)

    def recognize(self, faces: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for each face crop.
        Returns:
            np array (B, embedding_dim)
        """
        batch = self.preprocess(faces)
        if batch.size == 0:
            return np.zeros((0, self.model.buffer_shapes[self.model.primary_input][1] // (112*112*3)), dtype=np.float32)
        outputs = self.model.infer(
            {self.model.primary_input: batch}, batch.shape[0])
        # assume first output is embedding
        return list(outputs.values())[0]


class FacePipeline:
    """
    High-level pipeline: detect faces in images, draw boxes, save crops,
    extract embeddings, and report timings.
    """

    def __init__(self, detector_path: str, recognizer_path: str):
        """
        Initialize detector and recognizer.
        """
        self.detector = FaceDetector(detector_path)
        self.recognizer = FaceRecognizer(recognizer_path)

    def run(
            self,
            image_paths: List[str],
            max_batch_size: int = 32,
            output_dir: str = "output"
    ) -> None:
        """
        Execute full pipeline on a list of image paths.
        Saves annotated images and prints timing info.
        Args:
            image_paths: list of filesystem paths to images
            max_batch_size: number of images/faces per inference batch
            output_dir: directory to save annotated results
        """
        os.makedirs(output_dir, exist_ok=True)
        # load images with error checking
        images = []
        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                print(f"[WARN] Failed to read image: {p}")
                continue
            images.append(img)

        # detection loop
        all_boxes = []
        for i in range(0, len(images), max_batch_size):
            batch = images[i:i+max_batch_size]
            t0 = time.time()
            try:
                boxes = self.detector.detect_faces(batch)
            except Exception as e:
                print(
                    f"[ERROR] Detection failed on batch {i//max_batch_size}: {e}")
                boxes = [[] for _ in batch]
            t1 = time.time()
            print(
                f"[INFO] Detection batch {i//max_batch_size+1} of size {len(batch)} took {t1-t0:.3f}s")
            all_boxes.extend(boxes)
            del boxes
            gc.collect()

        # drawing & cropping
        crops = []
        for idx, (img, boxes) in enumerate(zip(images, all_boxes)):
            for (x1, y1, x2, y2, conf) in boxes:
                # draw bounding box and label
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{conf:.2f}", (x1, max(y1-10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # crop face region
                face = img[y1:y2, x1:x2]
                if face.size:
                    crops.append(face)
            # save annotated image
            out_path = os.path.join(output_dir, f"boxed_{idx+1}.jpg")
            cv2.imwrite(out_path, img)

        print(f"[INFO] Total faces detected: {len(crops)}")

        # recognition loop
        embeddings = []
        for i in range(0, len(crops), max_batch_size):
            batch = crops[i:i+max_batch_size]
            t0 = time.time()
            try:
                embs = self.recognizer.recognize(batch)
            except Exception as e:
                print(
                    f"[ERROR] Recognition failed on batch {i//max_batch_size}: {e}")
                embs = np.zeros((len(batch), 192), dtype=np.float32)
            t1 = time.time()
            print(
                f"[INFO] Recognition batch {i//max_batch_size+1} of size {len(batch)} took {t1-t0:.3f}s")
            embeddings.extend(embs)
            del embs, batch
            gc.collect()

        print(f"[INFO] Extracted {len(embeddings)} embeddings")


if __name__ == "__main__":
    # configure your engine paths and test images
    detector_path = "models/detection/trt-engine/scrfd_10g_gnkps_dynamic.engine"
    recognizer_path = "models/recognition/trt-engine/w600k_r50_dynamic.engine"
    # example list of images
    image_paths = [
        "test-images/image-1.jpg",
        "test-images/image-2.jpg",
        # add more...
    ] * 64

    pipeline = FacePipeline(detector_path, recognizer_path)
    start = time.time()
    pipeline.run(image_paths, max_batch_size=64, output_dir="output")
    print(f"[INFO] Total pipeline time: {time.time() - start:.3f}s")
