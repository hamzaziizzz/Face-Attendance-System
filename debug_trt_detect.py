# debug_trt_detect.py
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

ENGINE_PATH = "models/detection/trt-engine/scrfd_10g_gnkps.engine"

# 1) load engine & create context
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
with open(ENGINE_PATH, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine  = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# 2) find & set the one INPUT tensor shape
inp_name = None
for idx in range(engine.num_io_tensors):
    name = engine.get_tensor_name(idx)
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        inp_name = name
        context.set_input_shape(name, (1, 3, 640, 640))
        break
print("→ bound INPUT =", inp_name, "shape =", context.get_tensor_shape(inp_name))

# 3) allocate exactly one host/device buffer per tensor
host_bufs, dev_bufs = {}, {}
for idx in range(engine.num_io_tensors):
    name  = engine.get_tensor_name(idx)
    shape = context.get_tensor_shape(name)
    dtype = trt.nptype(engine.get_tensor_dtype(name))
    size  = int(np.prod(shape))
    h = cuda.pagelocked_empty(size, dtype)
    d = cuda.mem_alloc(h.nbytes)
    context.set_tensor_address(name, int(d))
    host_bufs[name] = h
    dev_bufs[name]  = d
    print(f"  tensor {name:15} mode={engine.get_tensor_mode(name).name:6} shape={shape} bytes={h.nbytes}")

# 4) load a test frame & preprocess exactly as your code does
frame = cv2.imread("/home/hamza/Pictures/IMG20240209191342-EDIT.jpg")  # pick a frame that has faces
frame = cv2.resize(frame, (640,360))
# letterbox to 640×640
pad = np.full((640,640,3), 114, dtype=np.uint8)
scale = min(640/ frame.shape[1], 640/ frame.shape[0])
nh, nw = int(frame.shape[0]*scale), int(frame.shape[1]*scale)
resized = cv2.resize(frame, (nw,nh))
top, left = (640-nh)//2, (640-nw)//2
pad[top:top+nh, left:left+nw] = resized
rgb = pad[..., ::-1].astype(np.float32)
norm = (rgb - 127.5)/128.0
inp = norm.transpose(2,0,1)[None, ...].ravel()

# 5) copy to device, run, copy back
# cuda.memcpy_htod(host_bufs[inp_name], inp)
# cuda.memcpy_htod(dev_bufs[inp_name], host_bufs[inp_name])

# copy your flattened input array directly into the device buffer:
cuda.memcpy_htod(dev_bufs[inp_name], inp)

# ret = context.execute_async_v3(cuda.Stream().handle)
# collect device pointers in binding order
bindings = [int(dev_bufs[name]) for name in host_bufs]
# synchronous execute_v2
success = context.execute_v2(bindings)
print("Sync execute_v2 →", success)

# now pull back all outputs
for name in host_bufs:
    if name == inp_name: continue
    cuda.memcpy_dtoh(host_bufs[name], dev_bufs[name])
    arr = host_bufs[name].reshape(context.get_tensor_shape(name))
    print(f"[OUT] {name:15} min={arr.min():.3f} max={arr.max():.3f} mean={arr.mean():.3f}")

