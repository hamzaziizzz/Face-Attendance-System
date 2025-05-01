#!/usr/bin/env python3
"""
batch_face_recognition_explained.py
===================================
A fully self‑documented, step‑by‑step version of the original *batch_face_recognition.py*.

Every single line is accompanied by an inline comment explaining **what** it does and
**why** it is necessary.
Variable names have been rewritten to be descriptive and self-explanatory.
Every public function/method carries a rich Google‑style docstring
(including argument/return‑type annotations) so that reading *only* the source is enough
to understand the entire pipeline.

Main building blocks
--------------------
1. **`TensorRTModel`** – Safe, reusable wrapper around a TensorRT engine.
2. **`FaceDetector`** – TensorRT‑accelerated SCRFD face detector with pre/post‑processing.
3. **`FaceRecognizer`** – TensorRT‑accelerated Arc‑Face recognizer producing embeddings.
4. **`FaceSearch`** – Milvus helper that turns embeddings → IDs.
5. **`FacePipeline`** – High‑level, batch‑friendly orchestration of the above blocks.

This rewrite is functionally identical to the original script but dramatically easier
to read, audit, and extend.
"""

import gc  # Trigger garbage collection between heavy batches
import math  # Mathematical helpers (e.g., square‑root, etc.)
# ──────────────────────────────────────────
# Standard‑library imports
# ──────────────────────────────────────────
import os  # Filesystem operations such as listdir() and path joins
import time  # Simple timing utilities for performance logging
from typing import List, Tuple, Dict, Any  # Static typing helpers

# Third‑party, non‑GPU utilities
import cv2  # OpenCV for image I/O + preprocessing
import numpy as np  # Numeric arrays / vectorized math
# GPU‑centric libraries
import pycuda.autoinit
import pycuda.driver as cuda  # Low‑level CUDA buffer management
import tensorrt as trt  # NVIDIA TensorRT for high‑throughput inference
# Vector‑database client
from pymilvus import connections, Collection  # Milvus search client + collection handler

# ──────────────────────────────────────────
# Module‑level constants – tweak these to taste
# ──────────────────────────────────────────
CONFIDENCE_THRESHOLD: float = 0.50  # Minimum detector confidence to keep a bbox
NMS_IOU_THRESHOLD: float = 0.40  # Maximum IoU overlap allowed during Non‑Max Suppression

# ──────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────

def letterbox_resize(
        input_image: np.ndarray,  # Raw BGR image straight from cv2.imread()
        target_shape: Tuple[int, int] = (640, 640),  # Desired network input (height, width)
        pad_color: Tuple[int, int, int] = (114, 114, 114)  # RGB value for padding bars
) -> Tuple[np.ndarray, float, int, int]:
    """Resize *and* pad an image, keeping its original aspect ratio.

    Args:
        input_image: Raw image in **BGR** format.
        target_shape: Desired *(height, width)*.
        Default to *(640, 640)* as required by SCRFD.
        pad_color: The color used for padded areas – default *(114, 114, 114)* matches
            Ultralytics/Yolo typical configuration.

    Returns:
        padded_image: The resized‑then‑padded image (shape *(H, W, 3)*).
        Scale_factor: `min(target_h / orig_h, target_w / orig_w)` applied during resize.
        pad_top:     Number of pixels added **above** the resized image.
        pad_left:    Number of pixels added **to the left** of the resized image.

    Raises:
        ValueError: If *input_image* is not a 3‑channel BGR image.
    """
    if input_image is None or input_image.ndim != 3 or input_image.shape[2] != 3:
        raise ValueError("Input must be a valid BGR image")  # Defensive programming

    orig_height, orig_width = input_image.shape[:2]  # Grab spatial dims (h, w)
    scale_factor: float = min(
        target_shape[0] / orig_height,  # How much can we scale by height?
        target_shape[1] / orig_width,  # …and by width?  Choose the smaller value.
    )

    # Compute a new, * scaled * resolution while preserving the aspect ratio
    new_width: int = int(orig_width * scale_factor)
    new_height: int = int(orig_height * scale_factor)

    resized_image: np.ndarray = cv2.resize(
        input_image,  # Raw frame
        (new_width, new_height),  # Target size (W, H)
        interpolation=cv2.INTER_LINEAR,  # Bilinear is good enough + inexpensive
    )

    # Compute padding required to fit exactly into *target_shape*
    pad_top: int = (target_shape[0] - new_height) // 2  # Pixels on top
    pad_left: int = (target_shape[1] - new_width) // 2  # Pixels on left

    # Create a solid canvas of the requested network resolution
    padded_image: np.ndarray = np.full(
        (target_shape[0], target_shape[1], 3),  # (H, W, C)
        pad_color,  # Uniform RGB/BGR color for letter‑box bars
        dtype=np.uint8,
    )

    # Paste the resized frame into the center of the canvas
    padded_image[pad_top : pad_top + new_height, pad_left : pad_left + new_width] = (
        resized_image
    )

    return padded_image, scale_factor, pad_top, pad_left  # Done!


def non_max_suppression(
        boxes_xyxy: np.ndarray,  # [[x1,y1,x2,y2], …] absolute pixel coords
        confidences: np.ndarray,  # Score per bbox (same length as *boxes_xyxy*)
        iou_threshold: float = NMS_IOU_THRESHOLD,  # IoU overlap allowed
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply standard *NMS* on a list of bounding boxes.

    Args:
        boxes_xyxy: *(N, 4)* array of corners *(x1, y1, x2, y2)*.
        confidences: *(N,)* array – higher ⇒ more likely a true face.
        iou_threshold: Any overlap **above** this value results in suppression.

    Returns:
        kept_boxes:  Bounding boxes **after** NMS.
        kept_scores: Their respective confidence scores.
    """

    if boxes_xyxy.size == 0:
        return boxes_xyxy, confidences  # Early‑exit – nothing to do

    # Split into convenient vectors for vectorized math
    x1, y1, x2, y2 = boxes_xyxy.T  # Each is shape *(N,)*
    areas = (x2 - x1) * (y2 - y1)  # Pre‑compute box areas for IoU calcs

    # Process boxes sorted by descending confidence (greedy NMS)
    order: np.ndarray = confidences.argsort()[::-1]  # Highest score first
    kept_idx: List[int] = []  # Indices of boxes *kept* by NMS

    while order.size:
        current: int = int(order[0])  # Index of highest‑score box
        kept_idx.append(current)  # Keep that one

        if order.size == 1:  # No more boxes to compare against
            break

        # Vectorised IoU between *current* and all *remaining* boxes
        xx1 = np.maximum(x1[current], x1[order[1:]])  # Left corner of overlap
        yy1 = np.maximum(y1[current], y1[order[1:]])  # Top corner of overlap
        xx2 = np.minimum(x2[current], x2[order[1:]])  # Right corner of overlap
        yy2 = np.minimum(y2[current], y2[order[1:]])  # Bottom corner of overlap
        intersect_w = np.maximum(0.0, xx2 - xx1)  # Clamp negative to 0
        intersect_h = np.maximum(0.0, yy2 - yy1)
        intersection = intersect_w * intersect_h  # Overlap area

        iou = intersection / (areas[current] + areas[order[1:]] - intersection + 1e-6)

        # Keep only *non‑overlapping* boxes (< threshold)
        remaining_mask = np.where(iou <= iou_threshold)[0]  # Indices in order[1:]
        order = order[remaining_mask + 1]  # +1 because we skipped order[0]

    return boxes_xyxy[kept_idx], confidences[kept_idx]

# ──────────────────────────────────────────
# TensorRT generic wrapper
# ──────────────────────────────────────────

class TensorRTModel:
    """Safe, reusable wrapper around a *single* TensorRT engine file."""

    def __init__(self, engine_path: str):
        """Load, deserialize, and prepare device buffers for a `.engine` file.

        Args:
            engine_path: Absolute or relative path to a TensorRT engine.
        """
        if not os.path.isfile(engine_path):  # Sanity‑check the path
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        self.logger: trt.Logger = trt.Logger(trt.Logger.WARNING)  # Less spam
        self.runtime: trt.Runtime = trt.Runtime(self.logger)  # Deserializer

        # ---- Engine deserialization ---------------------------------------------------
        with open(engine_path, "rb") as engine_file:
            engine_serialised: bytes = engine_file.read()  # Load the entire file into RAM
        self.engine: trt.ICudaEngine = self.runtime.deserialize_cuda_engine(
            engine_serialised
        )  # May raise RuntimeError if corrupt
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine → None returned")

        self.context: trt.IExecutionContext = self.engine.create_execution_context()
        self.cuda_stream: cuda.Stream = cuda.Stream()  # Async stream to overlap copies

        # ---- Discover I/O tensor names automatically ----------------------------------
        self.input_tensor_names: List[str] = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
               == trt.TensorIOMode.INPUT
        ]
        self.output_tensor_names: List[str] = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
               == trt.TensorIOMode.OUTPUT
        ]
        if not self.input_tensor_names or not self.output_tensor_names:
            raise RuntimeError("Engine has no detectable input/output tensors – abort")

        # ---- CUDA host/device buffers -------------------------------------------------
        # Will be (re‑)allocated lazily the first time `infer()` is called with a given
        # shape, and de‑allocated/re‑allocated automatically if the shape changes.
        self._host_buffers: Dict[str, np.ndarray] = {}
        self._device_buffers: Dict[str, cuda.DeviceAllocation] = {}
        self._cached_shapes: Dict[str, Tuple[int, ...]] = {}

    # Convenience property – assumes *single* network input
    @property
    def primary_input(self) -> str:  # → str (explicit return type optional in @property)
        """Return the first input tensor name (most networks have only one)."""
        return self.input_tensor_names[0]

    # ‑‑‑ Inference API ‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑‑
    def infer(
            self,
            host_inputs: Dict[str, np.ndarray],  # Maps tensor‑name → numpy host buffer
            batch_size: int,  # Only needed for networks with dynamic shape [0] dim
    ) -> Dict[str, np.ndarray]:
        """Run a *single* forward pass given a dict of input tensors.

        Args:
            host_inputs: Keyed by **engine tensor name**, containing ready‑to‑feed data.
            batch_size:  Size of the current mini‑batch (for dynamic‑shape networks).

        Returns:
            host_outputs: Dict that maps output‑tensor‑name → numpy array on host.

        Raises:
            ValueError: If *host_inputs* does not contain exactly the tensors the engine
                expects, or shapes do not match.
        """
        # 1) Shape validation + dynamic‑dim update --------------------------------------
        for tensor_name in self.input_tensor_names:
            if tensor_name not in host_inputs:
                raise ValueError(f"Missing mandatory input tensor: {tensor_name}")

            host_array: np.ndarray = host_inputs[tensor_name]
            self.context.set_input_shape(tensor_name, host_array.shape)  # TRT API call

        # 2) (Re‑)allocate buffers if shape changed -------------------------------------
        for tensor_name in self.input_tensor_names + self.output_tensor_names:
            current_shape: Tuple[int, ...] = tuple(self.context.get_tensor_shape(tensor_name))

            requested_bytes: int = int(np.prod(current_shape)) * np.dtype(trt.nptype(self.engine.get_tensor_dtype(tensor_name))).itemsize

            if tensor_name not in self._cached_shapes or current_shape != self._cached_shapes[tensor_name]:
                # Allocate page-locked host buffer (fast H2D/D2H)
                self._host_buffers[tensor_name] = cuda.pagelocked_empty(
                    int(np.prod(current_shape)), trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                )
                # Allocate/resize device buffer
                if tensor_name in self._device_buffers:
                    self._device_buffers[tensor_name].free()
                self._device_buffers[tensor_name] = cuda.mem_alloc(requested_bytes)
                self._cached_shapes[tensor_name] = current_shape

            # Bind device pointer to TRT execution context
            self.context.set_tensor_address(
                tensor_name, int(self._device_buffers[tensor_name])
            )

        # 3) **Host → Device** copies ---------------------------------------------------
        for tensor_name in self.input_tensor_names:
            np.copyto(self._host_buffers[tensor_name], host_inputs[tensor_name].ravel())
            cuda.memcpy_htod_async(
                self._device_buffers[tensor_name], self._host_buffers[tensor_name], self.cuda_stream
            )

        # 4) Execute network ------------------------------------------------------------
        self.context.execute_async_v3(self.cuda_stream.handle)  # Non‑blocking enqueue

        # 5) **Device → Host** copies ---------------------------------------------------
        for tensor_name in self.output_tensor_names:
            cuda.memcpy_dtoh_async(
                self._host_buffers[tensor_name], self._device_buffers[tensor_name], self.cuda_stream
            )
        self.cuda_stream.synchronize()  # Wait for all kernels + copies to finish

        # 6) Build outputs dict ---------------------------------------------------------
        host_outputs: Dict[str, np.ndarray] = {}
        for tensor_name in self.output_tensor_names:
            host_outputs[tensor_name] = self._host_buffers[tensor_name].reshape(
                self.context.get_tensor_shape(tensor_name)
            )
        return host_outputs

# ──────────────────────────────────────────
# Face detection wrapper (SCRFD) – preprocess / infer / postprocess
# ──────────────────────────────────────────

class FaceDetector:
    """Perform fast face detection using a TensorRT‑accelerated SCRFD model."""

    def __init__(
            self,
            detector_engine_path: str,  # Compiled SCRFD *.engine*
            network_input_hw: Tuple[int, int] = (640, 640),  # Standard SCRFD input shape
    ) -> None:
        self._trt_model: TensorRTModel = TensorRTModel(detector_engine_path)
        self.network_input_hw: Tuple[int, int] = network_input_hw
        self._strides: List[int] = [8, 16, 32]  # SCRFD produces three FPN scales

    # ‑‑‑ Helper: Pre‑processing ---------------------------------------------------------
    def _preprocess(
            self, raw_images_bgr: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[Tuple[int, int, float, int, int]]]:
        """Convert a list of BGR images into a **NCHW, float16** blob.

        Returns `(batched_blob, meta_list)` where *meta_list* keeps info required to
        reverse letter‑boxing after detection.
        """
        if not raw_images_bgr:
            raise ValueError("_preprocess() received an empty image list")

        batched_blob: List[np.ndarray] = []  # Will be stacked into shape *(B, 3, H, W)*
        meta_list: List[Tuple[int, int, float, int, int]] = []  # (orig_w,h,scale,pad_x,y)

        for img_bgr in raw_images_bgr:
            orig_height, orig_width = img_bgr.shape[:2]
            padded_image, scale, pad_top, pad_left = letterbox_resize(
                img_bgr, self.network_input_hw
            )

            # OpenCV → BGR; Network expects **RGB** scaled to [-1, 1]
            img_rgb = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB).astype(np.float16)
            img_normalised = (img_rgb - 127.5) / 128.0  # Range ~[-1,1]
            chw_blob = img_normalised.transpose(2, 0, 1)  # (H,W,C) → (C,H,W)

            batched_blob.append(chw_blob)
            meta_list.append((orig_width, orig_height, scale, pad_left, pad_top))

        return np.stack(batched_blob, axis=0), meta_list

    # ‑‑‑ Helper: Post‑processing --------------------------------------------------------
    def _postprocess(
            self,
            raw_outputs: Dict[str, np.ndarray],  # Direct TensorRT tensors
            meta_list: List[Tuple[int, int, float, int, int]],
    ) -> List[List[Tuple[int, int, int, int, float]]]:
        """Convert SCRFD raw tensors → human‑readable (x1,y1,x2,y2,score) per image."""
        batch_size: int = len(meta_list)

        # → Map tensor‑name → *(B, anchors, channels)* for convenience
        reshaped: Dict[str, np.ndarray] = {
            name: tensor.reshape(batch_size, -1, tensor.shape[-1] if tensor.ndim == 3 else 1)
            for name, tensor in raw_outputs.items()
        }

        all_detections: List[List[Tuple[int, int, int, int, float]]] = [
            [] for _ in range(batch_size)
        ]

        for img_idx in range(batch_size):
            decoded_boxes_list: List[np.ndarray] = []
            decoded_scores_list: List[np.ndarray] = []

            # SCRFD outputs three different FPN scales → iterate over them
            for stride in self._strides:
                scores = reshaped[f"score_{stride}"][img_idx].reshape(-1)
                bbox_deltas = reshaped[f"bbox_{stride}"][img_idx].reshape(-1, 4)

                # Each grid‑cell has two anchors → grid_size^2 * 2 = N anchors
                grid_side: int = int(math.sqrt(len(scores) // 2))
                grid_y, grid_x = np.meshgrid(
                    np.arange(grid_side), np.arange(grid_side), indexing="ij"
                )
                anchor_centres = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)
                anchor_centres = np.repeat(anchor_centres, 2, axis=0) * stride

                # Decode SCRFD boxes (l,t,r,b are *offsets* from the anchor center)
                left = bbox_deltas[:, 0] * stride
                top = bbox_deltas[:, 1] * stride
                right = bbox_deltas[:, 2] * stride
                bottom = bbox_deltas[:, 3] * stride

                x1 = anchor_centres[:, 0] - left
                y1 = anchor_centres[:, 1] - top
                x2 = anchor_centres[:, 0] + right
                y2 = anchor_centres[:, 1] + bottom

                conf_mask = scores > CONFIDENCE_THRESHOLD
                if conf_mask.any():
                    decoded_boxes_list.append(
                        np.stack([x1, y1, x2, y2], axis=1)[conf_mask]
                    )
                    decoded_scores_list.append(scores[conf_mask])

            # No faces in this frame → continue
            if not decoded_boxes_list:
                continue

            # Concatenate lists from the three different strides
            decoded_boxes = np.concatenate(decoded_boxes_list, axis=0)
            decoded_scores = np.concatenate(decoded_scores_list, axis=0)

            orig_w, orig_h, scale, pad_left, pad_top = meta_list[img_idx]

            # Undo letter‑boxing for each coordinate
            decoded_boxes[:, 0] = np.clip((decoded_boxes[:, 0] - pad_left) / scale, 0, orig_w)
            decoded_boxes[:, 1] = np.clip((decoded_boxes[:, 1] - pad_top) / scale, 0, orig_h)
            decoded_boxes[:, 2] = np.clip((decoded_boxes[:, 2] - pad_left) / scale, 0, orig_w)
            decoded_boxes[:, 3] = np.clip((decoded_boxes[:, 3] - pad_top) / scale, 0, orig_h)

            # Apply Non‑Max Suppression
            kept_boxes, kept_scores = non_max_suppression(decoded_boxes, decoded_scores)

            # Store results in friendly Python tuples
            for box, score in zip(kept_boxes, kept_scores):
                x1_i, y1_i, x2_i, y2_i = box.astype(int)
                all_detections[img_idx].append((x1_i, y1_i, x2_i, y2_i, float(score)))

        return all_detections

    # ‑‑‑ Public API ---------------------------------------------------------
    def detect_faces(
            self, images_bgr: List[np.ndarray]
    ) -> List[List[Tuple[int, int, int, int, float]]]:
        """Full detection pipeline: *preprocess → TensorRT → postprocess*."""
        blob, meta = self._preprocess(images_bgr)
        trt_outputs = self._trt_model.infer({self._trt_model.primary_input: blob}, blob.shape[0])
        return self._postprocess(trt_outputs, meta)

# ──────────────────────────────────────────
# Face recognition (ArcFace / w600k_r50)
# ──────────────────────────────────────────

class FaceRecognizer:
    """Generate 192‑D face embeddings using a TensorRT network."""

    def __init__(self, recogniser_engine_path: str) -> None:
        self._trt_model: TensorRTModel = TensorRTModel(recogniser_engine_path)

    # ‑‑‑ Helper: Pre‑processing ---------------------------------------------------------
    def _preprocess(self, face_crops: List[np.ndarray]) -> np.ndarray:
        """Resize to 112×112, normalize to [0,1], convert to NCHW float16 blob."""
        if not face_crops:
            # Return an *empty* (0,3,112,112) tensor to keep downstream logic simple
            return np.zeros((0, 3, 112, 112), dtype=np.float16)

        processed_list: List[np.ndarray] = []
        for crop in face_crops:
            if crop is None or crop.size == 0:
                continue  # Skip invalid crops
            resized = cv2.resize(crop, (112, 112)).astype(np.float16) / 255.0
            processed_list.append(resized.transpose(2, 0, 1))  # (C,H,W)

        return np.stack(processed_list) if processed_list else np.zeros((0, 3, 112, 112), dtype=np.float16)

    # ‑‑‑ Public API ---------------------------------------------------------
    def extract_embeddings(self, face_crops: List[np.ndarray]) -> np.ndarray:
        """Return a * (N, D)* matrix of float16 face embeddings."""
        blob = self._preprocess(face_crops)
        if blob.size == 0:
            return np.empty((0, 192), dtype=np.float16)  # 192‑D by default for w600k_r50

        trt_outputs = self._trt_model.infer({self._trt_model.primary_input: blob}, blob.shape[0])
        # Assume the *first* output is the feature vector
        return next(iter(trt_outputs.values()))

# ──────────────────────────────────────────
# Milvus‑based face search
# ──────────────────────────────────────────

class FaceSearch:
    """Simple helper that searches Milvus for nearest neighbors."""

    def __init__(
            self,
            host: str = "127.0.0.1",
            port: int = 19530,
            collection_name: str = "ABESIT_FACE_DATA_COLLECTION_FOR_COSINE",
    ) -> None:
        # Establish network connection (idempotent)
        connections.connect(host=host, port=port)
        self._collection: Collection = Collection(collection_name)
        self._collection.load()  # Must be loaded before `.search()`
        self._search_params: Dict[str, Any] = {"metric_type": "COSINE", "params": {"nprobe": 16}}

    def search(
            self, embedding_vectors: List[List[float]], limit: int = 1
    ) -> List[str | None]:
        """Return a list of *name_id* strings (or *None*) for each embedding vector."""
        try:
            results = self._collection.search(
                embedding_vectors,
                anns_field="embeddings",
                param=self._search_params,
                limit=limit,
                output_fields=["name_id"],
            )
            readable_results: List[str | None] = []  # Final user‑facing results
            for hits in results:
                if hits and hits[0].distance > 0.48:  # SCRFD expects distance = 1 - cosine
                    readable_results.append(hits[0].entity.get("name_id"))
                else:
                    readable_results.append(None)  # Low confidence → treat as unknown
            return readable_results
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[ERROR] Milvus search failed: {exc}")
            return [None] * len(embedding_vectors)

# ──────────────────────────────────────────
# End‑to‑end pipeline orchestrator
# ──────────────────────────────────────────

class FacePipeline:
    """High‑level batch pipeline: **detect → embed → search → annotate → save**."""

    def __init__(
            self,
            detector_engine_path: str,
            recogniser_engine_path: str,
            milvus_host: str = "localhost",
            milvus_port: int = 19530,
    ) -> None:
        self._detector = FaceDetector(detector_engine_path)
        self._recogniser = FaceRecognizer(recogniser_engine_path)
        self._searcher = FaceSearch(milvus_host, milvus_port)

    # ‑‑‑ PUBLIC API ---------------------------------------------------------
    def run(
            self,
            image_file_list: List[str],
            max_batch_size: int = 32,
            output_dir: str = "output",
    ) -> None:
        """End‑to‑end inference + annotation; writes images with rectangles to *output_dir*."""
        os.makedirs(output_dir, exist_ok=True)

        # ── 1) Load all images upfront ────────────────────────────────────────────
        raw_images: List[np.ndarray] = []
        for file_path in image_file_list:
            img = cv2.imread(file_path)
            if img is None:
                print(f"[WARN] Cannot decode image: {file_path}")
                continue
            raw_images.append(img)

        if not raw_images:
            print("[ERROR] No valid images found. Exiting pipeline.")
            return

        # ── 2) Detect faces in *mini‑batches* ─────────────────────────────────────
        all_detections: List[List[Tuple[int, int, int, int, float]]] = []
        for start_idx in range(0, len(raw_images), max_batch_size):
            batch_imgs = raw_images[start_idx : start_idx + max_batch_size]
            tic = time.time()
            detections = self._detector.detect_faces(batch_imgs)
            toc = time.time()
            print(
                f"[INFO] Detection batch {start_idx // max_batch_size + 1} (n={len(batch_imgs)}) completed in {toc - tic:.3f}s"
            )
            all_detections.extend(detections)
            gc.collect()

        print(f"[INFO] Total images processed for detection: {len(raw_images)}")

        # ── 3) Crop faces + keep mapping back to the parent image ─────────────────────
        face_crops: List[np.ndarray] = []
        crop_to_image_map: List[Dict[str, Any]] = []  # Each dict: {img_idx, bbox}

        for img_idx, (img, detections) in enumerate(zip(raw_images, all_detections)):
            for x1, y1, x2, y2, _ in detections:  # Score is not needed for crop
                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                face_crops.append(crop)
                crop_to_image_map.append({"img_idx": img_idx, "bbox": (x1, y1, x2, y2)})

        if not face_crops:
            print("[ERROR] No faces detected across entire dataset.")
            return

        print(f"[INFO] Total face crops to recognise: {len(face_crops)}")

        # ── 4 & 5) Recognise + Milvus search in batches ──────────────────────────
        for start_idx in range(0, len(face_crops), max_batch_size):
            batch_crops = face_crops[start_idx : start_idx + max_batch_size]
            batch_map = crop_to_image_map[start_idx : start_idx + max_batch_size]

            tic = time.time()
            embeddings = self._recogniser.extract_embeddings(batch_crops)
            toc = time.time()
            print(
                f"[INFO] Recognition batch {start_idx // max_batch_size + 1} (n={len(batch_crops)}) in {toc - tic:.3f}s"
            )

            tic = time.time()
            name_ids = self._searcher.search(embeddings.tolist())
            toc = time.time()
            print(
                f"[INFO] Milvus search batch {start_idx // max_batch_size + 1} (n={len(batch_crops)}) in {toc - tic:.3f}s"
            )

            # ── 6) Annotate parent images with the search results ────────────────
            for mapping, name_id in zip(batch_map, name_ids):
                img_idx = mapping["img_idx"]
                x1, y1, x2, y2 = mapping["bbox"]
                label = name_id if name_id is not None else "Unknown"

                # Draw rectangle + label (OpenCV mutates in‑place)
                cv2.rectangle(raw_images[img_idx], (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    raw_images[img_idx],
                    label,
                    (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )

            gc.collect()  # Free VRAM/host RAM sooner rather than later

        # ── 7) Persist annotated images to *output_dir* ──────────────────────────
        for idx, annotated_img in enumerate(raw_images):
            save_path = os.path.join(output_dir, f"boxed_{idx + 1}.jpg")
            cv2.imwrite(save_path, annotated_img)
        print(f"[INFO] All annotated images written to: {output_dir}")

# ──────────────────────────────────────────
# Example *main* entry‑point (same as original)
# ──────────────────────────────────────────

if __name__ == "__main__":
    DETECTOR_ENGINE = "models/detection/trt-engine/scrfd_10g_gnkps_dynamic.engine"  # SCRFD
    RECOGNISER_ENGINE = "models/recognition/trt-engine/w600k_r50_dynamic.engine"  # ArcFace

    # Example directory with test JPGs – change at will
    INPUT_DIRECTORY = "/home/hamza/OfficeProjects/Face-Super-Resolution/2023CSDS162"
    OUTPUT_DIRECTORY = os.path.basename(INPUT_DIRECTORY.rstrip(os.sep))  # Use folder name

    # Gather *.jpg files into a list
    image_files: List[str] = [
        os.path.join(INPUT_DIRECTORY, f) for f in os.listdir(INPUT_DIRECTORY) if f.lower().endswith(".jpg")
    ]

    pipeline = FacePipeline(DETECTOR_ENGINE, RECOGNISER_ENGINE)
    tic = time.time()
    pipeline.run(image_files, max_batch_size=64, output_dir=OUTPUT_DIRECTORY)
    toc = time.time()
    print(f"[INFO] Full pipeline completed in {toc - tic:.2f} seconds")
