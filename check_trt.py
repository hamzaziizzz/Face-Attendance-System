import tensorrt as trt

# Path to your engine
engine_path = "models/detection/trt-engine/scrfd_10g_gnkps_dynamic.engine"

# Create logger
logger = trt.Logger(trt.Logger.WARNING)

# Load the engine
with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

    print(f"Number of bindings: {engine.num_io_tensors}")

    # for idx in range(engine.num_io_tensors):
    #     name = engine.get_tensor_name(idx)
    #     dtype = engine.get_tensor_dtype(idx)
    #     shape = engine.get_tensort_shape(idx)

    input_names = [
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i)) ==
           trt.TensorIOMode.INPUT
    ]
    output_names = [
        engine.get_tensor_name(i)
        for i in range(engine.num_io_tensors)
        if engine.get_tensor_mode(engine.get_tensor_name(i)) ==
           trt.TensorIOMode.OUTPUT
    ]

    print(f"Input names: {input_names}")
    print(f"Output names: {output_names}")
