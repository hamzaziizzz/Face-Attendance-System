"""
tensorrt_inference_explained.py
================================
Fully‑annotated, self‑contained face‑detection + face‑recognition pipeline
implemented with NVIDIA TensorRT, PyCUDA for GPU memory management, OpenCV
for video IO / drawing, and Milvus for similarity search.

Every single line is followed by an inline comment that explains **exactly**
what that line does.
Function and class doc‑strings clearly describe the
purpose, arguments, return types, and side‑effects so that the script can be
read and understood without additional documentation.

Author : ChatGPT—refactored and documented on 29‑Apr‑2025
"""

# ── Standard‑library imports ────────────────────────────────────────────────
import math                                   # Maths utilities (sqrt, ceil …)
import time                                   # Timing / FPS calculation
import queue                                  # Thread‑safe FIFO for frame pass
import threading                              # For running each RTSP stream
from typing import List, Tuple                # Type hints for readability

# ── Third‑party imports (CPU side) ──────────────────────────────────────────
import cv2                                    # OpenCV for image & video IO
import numpy as np                            # ndarray maths & manipulation
import pycuda
import tensorrt as trt                        # Deep‑learning runtime
import pycuda.driver as cuda                  # GPU memory access and transfer

# Initialize the CUDA driver once at import time (no *pycuda.autoinit*)
cuda.init()                                   # Must be called before any CUDA

from pymilvus import connections, Collection  # Vector DB for face search
# ────────────────────────────────────────────────────────────────────────────


# ────────────────────── TensorRT helper utilities ──────────────────────────
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)   # Logger instance (show warnings)


def load_engine(engine_path: str) -> trt.ICudaEngine:
    """
    Deserialize a *.engine* file from the disk into a live TensorRT Engine.

    Parameters
    ----------
    engine_path : str
        Filesystem path pointing to the TensorRT engine file.

    Returns
    -------
    trt.ICudaEngine
        The deserialized, ready‑to‑run engine.
    """
    with open(engine_path, "rb") as engine_file:                      # Read the file
        runtime = trt.Runtime(TRT_LOGGER)                             # Create RT
        return runtime.deserialize_cuda_engine(engine_file.read())    # -> engine


def allocate_io_buffers(
        execution_ctx: trt.IExecutionContext,
        engine: trt.ICudaEngine
) -> Tuple[
    List[Tuple[np.ndarray, "pycuda._driver.DeviceAllocation"]],
    List[Tuple[np.ndarray, "pycuda._driver.DeviceAllocation"]],
    List[int]
]:
    """
    Allocate pinned (page‑locked) host buffers + device buffers for every
    tensor in the network and inform TensorRT where each tensor lives on the
    GPU.
    Returns separate *input_buffers*, *output_buffers*, plus the raw
    `bindings` list needed by `context.execute_v2`.

    Notes
    -----
    * **Pinned host memory** is mandatory for asynchronous, fast transfer.
    * The function supports both explicit and implicit batch models.

    Parameters
    ----------
    execution_ctx : trt.IExecutionContext
        The active execution context which owns the tensor addresses.
    engine : trt.ICudaEngine
        The compiled engine (needed for tensor metadata).

    Returns
    -------
    Tuple[inputs, outputs, bindings]
    """
    input_buffers:  List[Tuple[np.ndarray, cuda.DeviceAllocation]] = []  # host+dev
    output_buffers: List[Tuple[np.ndarray, cuda.DeviceAllocation]] = []  # host+dev
    bindings: List[int] = []                                             # device ptrs

    # Iterate over every network IO tensor (both input and output)
    for tensor_idx in range(engine.num_io_tensors):
        tensor_name: str = engine.get_tensor_name(tensor_idx)            # Tensor id
        tensor_shape = execution_ctx.get_tensor_shape(tensor_name)       # e.g. (1,3,640,640)
        tensor_dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))  # np.float32
        num_elems: int = int(np.prod(tensor_shape))                      # Total elems

        host_mem = cuda.pagelocked_empty(num_elems, tensor_dtype)        # Pinned RAM
        device_mem = cuda.mem_alloc(host_mem.nbytes)                     # GPU buffer

        execution_ctx.set_tensor_address(tensor_name, int(device_mem))   # Tell TRT
        bindings.append(int(device_mem))                                 # For execute_v2

        # Separate lists by IO direction for convenience during inference
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            input_buffers.append((host_mem, device_mem))
        else:
            output_buffers.append((host_mem, device_mem))

    return input_buffers, output_buffers, bindings
# ────────────────────────────────────────────────────────────────────────────


# ──────────────────────────── Face Detector (TRT) ───────────────────────────
class FaceDetectorTRT:
    """
    SCRFD‑based face detector sped up with TensorRT.

    The detector expects **letter‑boxed**, RGB‑ordered, float32 inputs in the
    `[0,1]` ranges (after (x‑0.5)/0.5 normalization) and returns a list of
    bounding boxes in the original image coordinate space along with a
    confidence score.

    Attributes
    ----------
    model_input_size : Tuple[int, int]
        Width × Height passed to the network (must match engine build).
    confidence_threshold : float
        Minimum score for a proposal to be kept before NMS.
    nms_threshold : float
        IoU threshold for Non‑Maximum Suppression.
    feature_map_strides : List[int]
        Strides that correspond to the output feature maps (SCRFD uses 8/16/32).
    engine : trt.ICudaEngine
        The deserialized TensorRT engine.
    context : trt.IExecutionContext
        Execution context bound to the engine.
    Inputs / outputs / bindings : see `allocate_io_buffers`.
    """

    def __init__(
            self,
            engine_path: str,
            model_input_size: Tuple[int, int] = (640, 640),
            confidence_threshold: float = 0.50,
            nms_threshold: float = 0.40
    ) -> None:
        # Save hyper‑parameters
        self.model_input_size = model_input_size          # (W, H) used by the model
        self.confidence_threshold = confidence_threshold  # filter low scores
        self.nms_threshold = nms_threshold                # IoU cutoff for NMS
        self.feature_map_strides = [8, 16, 32]            # SCRFD default

        # Deserialize the TRT engine and create an execution context
        self.engine = load_engine(engine_path)            # compiled model
        self.context = self.engine.create_execution_context()  # runtime ctx

        # Tell TensorRT the *dynamic* input shape (if applicable)
        for tensor_idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(tensor_idx)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, (1, 3, *self.model_input_size))
                break  # assume a single input

        # Pre‑allocate buffers once (host + GPU) for maximal performance
        (self.inputs,
         self.outputs,
         self.bindings) = allocate_io_buffers(self.context, self.engine)

    # ─────────────────────── Pre‑processing helpers ────────────────────────
    def preprocess(
            self,
            bgr_img: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[float, int, int]]:
        """
        Resize + pad BGR image to model size, convert to normalised RGB blob.

        Parameters
        ----------
        bgr_img : np.ndarray
            Original OpenCV frame in BGR color order.

        Returns
        -------
        Tuple
            (blob, # Flattened float32 array ready for TRT upload
                (orig_h, orig_w),
                (scale, pad_top, pad_left))
        """
        orig_h, orig_w = bgr_img.shape[:2]                                           # Keep for mapping back
        target_w, target_h = self.model_input_size                                   # Desired dims (W,H)
        scale = min(target_w / orig_w, target_h / orig_h)                            # Uniform scaling
        new_h, new_w = int(orig_h * scale), int(orig_w * scale)                      # Scaled size

        resized = cv2.resize(bgr_img, (new_w, new_h), interpolation=cv2.INTER_AREA)  # Resize keeping aspect
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)               # Letter‑box canvas

        pad_top, pad_left = (target_h - new_h) // 2, (target_w - new_w) // 2         # Center placement
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized         # Insert resized ROI

        rgb_img = padded[..., ::-1].astype(np.float32)                               # BGR -> RGB + float32
        normalised = (rgb_img - 127.5) / 128.0                                       # ~[-1,1] range

        blob = normalised.transpose(2, 0, 1)[None, ...].ravel()                      # CHW & flatten

        return blob, (orig_h, orig_w), (scale, pad_top, pad_left)

    def infer(self, input_blob: np.ndarray) -> "list[np.ndarray]":
        """Run a forward pass on the detector network and return raw outputs.

        Parameters
        ----------
        input_blob : np.ndarray
            Flattened CHW float32 array produced by:             meth:`FaceDetectorTRT.preprocess`.

        Returns
        -------
        list[np.ndarray]
            A list where each item is a **copy** of one output tensor residing on
            host memory, in the same order that TensorRT exposes them.
        """
        # Retrieve the sole input buffer pair (pinned host array + device ptr)
        host_in, device_in = self.inputs[0]

        # Copy the prepared blob into the pinned host buffer.
        # Using np.copyto ensures efficient memory move with dtype checks.
        np.copyto(host_in, input_blob)

        # Upload from host → device (GPU) synchronously.
        # Since the host memory is *page‑locked*, the transfer is DMA‑optimized.
        cuda.memcpy_htod(device_in, host_in)

        # Kick off inference.
        # `execute_v2` takes the *bindings* list which contains raw device addresses for *all* network tensors.
        if not self.context.execute_v2(self.bindings):
            raise RuntimeError("TensorRT execute_v2 failed for FaceDetectorTRT")

        # Collect model outputs.
        # Each element in `self.outputs` is a tuple host_array, device_allocation).
        # We copy *device → host* for every output, then append *a copy* to `results` so that later calls
        # don’t overwrite previously returned data.
        results: list[np.ndarray] = []
        for host_out, device_out in self.outputs:
            cuda.memcpy_dtoh(host_out, device_out)
            results.append(host_out.copy())  # Important: copy to detach from the buffer

        return results

    # ───────────────────────── IoU & NMS utilities ─────────────────────────
    @staticmethod
    def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Vectorized Intersection‑over‑Union between *box* and many *boxes*."""
        x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])

        inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        return inter_area / (box_area + boxes_area - inter_area + 1e-6)

    def postprocess(
            self,
            raw_outputs: List[np.ndarray],
            original_shape: Tuple[int, int],
            scale_info: Tuple[float, int, int]
    ) -> List[Tuple[int, int, int, int, float]]:
        """
        Convert raw network outputs into *pixel‑accurate* bounding boxes
        referenced to the **original image coordinate system**.

        Parameters
        ----------
        raw_outputs : List[np.ndarray]
            Model outputs as copied from the GPU (order depends on ONNX export).
        original_shape : Tuple[int, int]
            (height, width) of the image before any resizing.
        scale_info : Tuple[float, int, int]
            (scale, pad_top, pad_left) returned by `preprocess`.

        Returns
        -------
        List[Tuple[x1, y1, x2, y2, score]]
            Bounding boxes + probability after NMS.
        """
        orig_h, orig_w = original_shape
        scale, pad_top, pad_left = scale_info

        confidences: List[np.ndarray] = []     # Hold scores per stride
        decoded_boxes: List[np.ndarray] = []   # Hold decoded box coords

        num_strides = len(self.feature_map_strides)

        # SCRFD packs class‑scores first followed by bbox deltas for each stride
        for stride_idx, stride_val in enumerate(self.feature_map_strides):
            score_tensor = raw_outputs[stride_idx].reshape(-1)                 # (anchors,)
            delta_tensor = raw_outputs[stride_idx + num_strides].reshape(-1, 4)# (anchors,4)

            anchors_per_cell = 2                                               # SCRFD spec
            num_positions = delta_tensor.shape[0] // anchors_per_cell
            grid_side = int(math.sqrt(num_positions))                          # feature‑map dim

            # Build anchor center coordinates (x,y), then repeat per anchor type
            ys, xs = np.meshgrid(range(grid_side), range(grid_side), indexing="ij")
            centres = np.stack((xs, ys), axis=-1).reshape(-1, 2) * stride_val
            centres = np.repeat(centres, anchors_per_cell, axis=0)

            # Decode left/top/right/bottom distances to corner coords
            lefts, tops, rights, bottoms = [delta_tensor[:, j] * stride_val for j in range(4)]
            x1 = centres[:, 0] - lefts;  y1 = centres[:, 1] - tops
            x2 = centres[:, 0] + rights; y2 = centres[:, 1] + bottoms

            mask = score_tensor > self.confidence_threshold                    # Preliminary filter
            if mask.any():
                decoded_boxes.append(np.stack([x1, y1, x2, y2], axis=1)[mask])  # store box coordinates
                confidences.append(score_tensor[mask])  # store corresponding confidence scores

        # ── Early exit: if no boxes passed the confidence threshold ──────────
        if not decoded_boxes:
            return []

        # Concatenate list elements into single NDArrays for vectorized ops
        all_boxes = np.vstack(decoded_boxes)       # shape ⇒ (N, 4)
        all_scores = np.hstack(confidences)        # shape ⇒ (N,)

        # ── Non‑Maximum Suppression (greedy implementation) ─────────────────
        order = np.argsort(all_scores)[::-1]       # Indices sorted by score (desc)
        keep: List[int] = []                       # Indices that survive NMS

        while order.size:
            current = order[0]                     # Highest‑scoring box
            keep.append(int(current))
            if order.size == 1:                    # Nothing left to compare
                break
            ious = self._iou(all_boxes[current], all_boxes[order[1:]])
            order = order[1:][ious < self.nms_threshold]  # Drop boxes with IoU ≥ thresh

        # ── Re‑map surviving boxes back to original image coordinates ───────
        results: List[Tuple[int, int, int, int, float]] = []
        for idx in keep:
            bx1, by1, bx2, by2 = all_boxes[idx]
            # Undo letter‑box scaling + padding
            x1 = int(max(0, min(orig_w, (bx1 - pad_left) / scale)))
            y1 = int(max(0, min(orig_h, (by1 - pad_top)  / scale)))
            x2 = int(max(0, min(orig_w, (bx2 - pad_left) / scale)))
            y2 = int(max(0, min(orig_h, (by2 - pad_top)  / scale)))
            results.append((x1, y1, x2, y2, float(all_scores[idx])))

        return results
# ────────────────────────────────────────────────────────────────────────────



# ─────────────────────────── Face Recognizer (TRT) ──────────────────────────
class FaceRecognizerTRT:  # pylint: disable=too-few-public-methods
    """ArcFace‑style face embedding extractor sped up with TensorRT."""

    # ---------------------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------------------
    def __init__(self, engine_path: str) -> None:
        """Load the TensorRT engine and prepare IO buffers.

        Parameters
        ----------
        engine_path : str
            Filesystem path to the serialized TensorRT engine (.engine).
        """
        # Deserialize the `.engine` file into a CUDA‑executable network
        self.engine = load_engine(engine_path)  # trt.ICudaEngine instance

        # Create an *execution context* – this holds binding shapes,
        # optimization profile selection, and state needed for inference.
        self.context = self.engine.create_execution_context()

        # Inform TensorRT about the expected shape of the *input* tensor.
        # ArcFace expects a **single RGB face crop** of size 112×112.
        # We iterate over all network IO tensors until we find the first one
        # flagged as `INPUT`, then fix its shape to (N=1,C=3,H=112,W=112).
        for tensor_index in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(tensor_index)  # e.g. 'input.1'
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                # Set explicit batch‑dim (1) and spatially dims (3×112×112)
                self.context.set_input_shape(tensor_name, (1, 3, 112, 112))
                break  # Only one input tensor expected, so we can exit early.

        # Allocate **pinned host memory** and **device buffers** for every
        # network tensor and store their addresses inside the context so
        # that further calls to `execute_v2()` know where to read/write.
        (
            self.input_buffers,   # List[(host_ndarray, device_ptr)] for inputs
            self.output_buffers,  # List[(host_ndarray, device_ptr)] for outputs
            self.bindings         # List[int] of *all* device pointers ordered
        ) = allocate_io_buffers(self.context, self.engine)

    # ---------------------------------------------------------------------
    # Inference method
    # ---------------------------------------------------------------------
    def infer(self, face_blob: np.ndarray) -> np.ndarray:
        """Run forward pass and return the face embedding.

        Parameters
        ----------
        face_blob : np.ndarray
            Pre‑processed tensor with shape **(1, 3, 112, 112)** & *float32*
            dtype in the **[-1, 1]** range.
            This is exactly what:             func:`preprocess_for_recognition` generates.

        Returns
        -------
        np.ndarray
            A 2‑D array with shape **(1, F)** where *F* is the embedding size
            (commonly 512).
            Each row is L2‑normalized by the network.
        """
        # Retrieve the first (and only) *input* host/GPU buffer pair.
        host_input, device_input = self.input_buffers[0]

        # Pre-process the face crop from raw BGR to a flattened face blob
        face_blob = self._preprocess(face_blob)

        # Copy the flattened blob into the *pinned* host memory region.
        # `np.copy` is faster than simple slicing when dtypes match.
        np.copyto(host_input, face_blob.ravel())

        # Upload host → GPU (synchronous).
        # Because the host buffer is
        # *page‑locked*, this transfer can use DMA for maximum speed.
        cuda.memcpy_htod(device_input, host_input)

        # Launch inference.
        # `execute_v2()` consumes the *bindings* list we
        # prepared earlier which contains the raw device addresses.
        if not self.context.execute_v2(self.bindings):
            # If TensorRT returns `False`, something went wrong during kernel
            # execution.
            # We raise an error so the caller can handle it.
            raise RuntimeError("TensorRT execute_v2 failed for FaceRecognizerTRT")

        # Download the output embedding back to host memory.
        host_output, device_output = self.output_buffers[0]
        cuda.memcpy_dtoh(host_output, device_output)  # GPU → Host transfer

        # Reshape to (1, -1) so that callers don't have to remember the
        # exact embedding length.
        return host_output.reshape(1, -1)

    # ---------------------------------------------------------------------
    # Preprocess method
    # ---------------------------------------------------------------------
    @staticmethod
    def _preprocess(face_bgr: np.ndarray) -> np.ndarray:  # noqa: D401
        """Convert a raw **BGR** face crop into a normalized tensor-suitable
        face region in OpenCV's BGR color space, typically sliced from
            or the ArcFace backbone.

        This helper is called internally by: meth:`infer` and should not be
        used directly by external code.

        Parameters
        ----------
        face_bgr : np.ndarray
            Original video frame.

        Returns
        -------
        np.ndarray
            Float32 tensor with shape ``(1, 3, 112, 112)`` and value range
            ``[-1, 1]`` ready to be copied to GPU memory.
        """

        # Convert from OpenCV's default BGR to RGB because ArcFace was trained
        # on RGB images.
        # The color swap is done via `cv2.cvtColor,` which is
        # highly optimized and supports SIMD under the hood.
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # Resize the crop to the canonical **112×112** spatial resolution using
        # *area* interpolation, which is well suited for downscaling and avoids
        # aliasing artifacts.
        resized = cv2.resize(face_rgb, (112, 112), interpolation=cv2.INTER_AREA)

        # Convert to `float32` then scale pixel values from *[0,255]* to *[0,1]*.
        resized_f32 = resized.astype(np.float32) / 255.0

        # Apply mean‑0, range‑[-1,1] normalization expected by the model.
        normalised = (resized_f32 - 0.5) / 0.5

        # Re‑order dimensions from HWC→CHW and add a *batch* dimension so that
        # the tensor shape becomes **(N=1, C=3, H=112, W=112)** exactly as we
        # configured in `__init__()`.
        blob = normalised.transpose(2, 0, 1)[None, ...]

        # Return the prepared tensor – no further flattening is required here
        # because the `infer()` method will internally flatten the array when
        # copying to the pinned host buffer.
        return blob
# ────────────────────────────────────────────────────────────────────────────



def stream_worker_verbose(
        stream_source,
        frame_queue: "queue.Queue[np.ndarray]",
        detector_engine_path: str,
        recogniser_engine_path: str,
        milvus_collection: Collection,
        milvus_search_params: dict,
) -> None:
    """Thread target that performs end‑to‑end detection + recognition.

    This *verbose* flavor contains exhaustive inline comments explaining every
    single step, from grabbing frames all the way to pushing the annotated
    result into a queue that the main thread consumes for display.

    Parameters
    ----------
    stream_source: (int | str)
        Can be a webcam index (e.g. ``0``), a file path, or an RTSP/HTTP URL.
    frame_queue : queue.Queue[np.ndarray]
        Thread‑safe buffer where the latest annotated frame is published.
    detector_engine_path : str
        Path to the TensorRT engine for **SCRFD** face detection.
    recogniser_engine_path : str
        Path to the TensorRT engine for **ArcFace** face recognition.
    milvus_collection : Collection
        Pre‑loaded Milvus collection containing facial embeddings.
    milvus_search_params : dict
        Dictionary of search parameters (metric, nprobe, …) for Milvus.
    """

    # Each thread needs its **own** CUDA primary context when using
    # PyCUDA.
    # Creating it at the top ensures all following CUDA calls
    # (memory alloc, memcpy, kernel launch) are associated with *this*
    # thread rather than the main Python thread.
    cuda_context = cuda.Device(0).make_context(0)

    try:
        # Deserialize the TensorRT engines once per thread.
        # Loading inside the thread avoids needless contention and gives each
        # stream its own execution context (no mutex required).
        detector = FaceDetectorTRT(detector_engine_path)
        recogniser = FaceRecognizerTRT(recogniser_engine_path)

        # OpenCV VideoCapture works for cameras *and* URLs.
        # Automatic back‑end selection takes care of decoding.
        capture = cv2.VideoCapture(stream_source)

        frame_counter = 0
        timer_start = time.time()
        instantaneous_fps = 0.0

        # Main processing loop -------------------------------------------------
        while capture.isOpened():
            ok, frame = capture.read()        # Grab the next frame
            if not ok:
                break                            # End of stream / error
            frame_bgr = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
            original_h, original_w = frame_bgr.shape[:2]

            # *Detection* on a smaller preview to speed up inference ----
            preview = cv2.resize(frame_bgr, (640, 360), interpolation=cv2.INTER_AREA)
            blob, orig_shape, scale_meta = detector.preprocess(preview)
            raw_det = detector.infer(blob)
            detections = detector.postprocess(raw_det, orig_shape, scale_meta)

            # *Recognition* for each detected face ------------------------
            for px1, py1, px2, py2, det_score in detections:
                # Map preview coordinates back to full‑resolution frame
                scale_x, scale_y = original_w / 640, original_h / 360
                fx1, fy1 = int(px1 * scale_x), int(py1 * scale_y)
                fx2, fy2 = int(px2 * scale_x), int(py2 * scale_y)

                face_crop = frame_bgr[fy1:fy2, fx1:fx2]
                if face_crop.size == 0:
                    continue  # Skip degenerate boxes

                # Preprocess & embed
                embedding = recogniser.infer(face_crop)[0].tolist()

                # Query Milvus for the nearest neighbor (cosine similarity)
                search_result = milvus_collection.search(
                    [embedding],
                    anns_field="embeddings",
                    param=milvus_search_params,
                    limit=1,
                    output_fields=["name_id"],
                )

                # Decide to draw color/label based on the score threshold
                box_colour = (0, 0, 255)  # Red by default (no match)
                label_text = ""
                if search_result and len(search_result[0]) > 0:
                    top_hit = search_result[0][0]
                    similarity = top_hit.score  # 0‑1 cosine similarity
                    identity = top_hit.entity.get("name_id")
                    if similarity >= 0.48:
                        box_colour = (0, 255, 0)  # Green if confident match
                    label_text = f"{identity}: {similarity:.2f}"

                # Draw bounding box + label
                cv2.rectangle(frame_bgr, (fx1, fy1), (fx2, fy2), box_colour, 2)
                if label_text:
                    cv2.putText(
                        frame_bgr,
                        label_text,
                        (fx1, fy1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        box_colour,
                        2,
                    )

            # FPS counter ---------------------------------------------------
            frame_counter += 1
            elapsed = time.time() - timer_start
            if elapsed >= 1.0:
                instantaneous_fps = frame_counter / elapsed
                frame_counter = 0
                timer_start = time.time()
            cv2.putText(
                frame_bgr,
                f"FPS: {instantaneous_fps:.2f}",
                (10, original_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 255),
                2,
            )

            # Publish frame (dropping stale ones) ---------------------------
            if frame_queue.full():
                try:
                    frame_queue.get_nowait()
                except queue.Empty:
                    pass  # Queue already consumed
            frame_queue.put(frame_bgr)

        # Release resources on exit ----------------------------------------
        capture.release()
        cv2.destroyAllWindows()
        cuda_context.pop()

    finally:
        cv2.destroyAllWindows()
        # Always pop the CUDA context to avoid leaks when the thread exits.
        cuda_context.pop()


# ---------------------------------------------------------------------------
# Super‑verbose `main` that spawns workers and draws a mosaic window.
# ---------------------------------------------------------------------------

def main_verbose() -> None:
    """Program entry‑point: starts worker threads and displays live mosaic."""

    # A) — Connect to Milvus and load collection *once* ---------------------
    connections.connect(alias="default", host="192.168.12.1", port="19530")
    collection = Collection("ABESIT_FACE_DATA_COLLECTION_FOR_COSINE")
    collection.load()  # Ensure data is in memory

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

    # B) — Define the list of sources.  Replace/extend as needed. ---------------
    sources = [
        "rtsp://grilsquad:grilsquad@192.168.12.17/554/stream1",
    ]  # Webcam index 0 by default

    # C) — Spawn one worker thread per source -------------------------------
    queues_by_src: dict = {}
    for src in sources:
        q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1)
        t = threading.Thread(
            target=stream_worker_verbose,
            args=(
                src,
                q,
                "models/detection/trt-engine/scrfd_10g_gnkps.engine",
                "models/recognition/trt-engine/w600k_r50.engine",
                collection,
                search_params,
            ),
            daemon=True,
        )
        t.start()
        queues_by_src[src] = q

    # D) — Prepare mosaic canvas geometry ----------------------------------
    tile_h, tile_w = 720, 1280
    n_streams = len(sources)
    cols = math.ceil(math.sqrt(n_streams))
    rows = math.ceil(n_streams / cols)

    cv2.namedWindow("FaceRec TRT Mosaic (verbose)", cv2.WINDOW_NORMAL)

    # Placeholder black frames
    last_frame = {
        s: np.zeros((tile_h, tile_w, 3), dtype=np.uint8) for s in sources
    }

    # E) — Main display loop ------------------------------------------------
    try:
        while True:
            thumbnails: List[np.ndarray] = []

            for s in sources:
                if not queues_by_src[s].empty():
                    last_frame[s] = cv2.resize(
                        queues_by_src[s].get(),
                        (tile_w, tile_h),
                        interpolation=cv2.INTER_AREA,
                    )
                thumbnails.append(last_frame[s])

            # Assemble mosaic grid
            mosaic = np.zeros((tile_h * rows, tile_w * cols, 3), dtype=np.uint8)
            for idx, thumb in enumerate(thumbnails):
                r, c = divmod(idx, cols)
                mosaic[
                r * tile_h : (r + 1) * tile_h,
                c * tile_w : (c + 1) * tile_w,
                ] = thumb

            cv2.imshow("FaceRec TRT Mosaic (verbose)", mosaic)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break  # Graceful exit on 'q'

            time.sleep(0.01)  # Throttle to reduce CPU usage

        cv2.destroyAllWindows()
    finally:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Run the verbose version if this file is executed as a script.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main_verbose()
