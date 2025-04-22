import os
import sys
import argparse
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def build_engine(onnx_path, engine_path, fp16=False, max_batch_size=1, max_workspace_size=1024):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Set workspace size using new API for TensorRT 10+
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            max_workspace_size * (1 << 20)
        )

        # FP16 precision flag
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] FP16 precision enabled.")

        # Parse ONNX model
        with open(onnx_path, 'rb') as model_file:
            if not parser.parse(model_file.read()):
                print("[ERROR] Failed to parse ONNX model.")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                sys.exit(1)

        # Create optimization profile
        # input_tensor = network.get_input(0)
        # shape = list(input_tensor.shape)
        # shape[0] = 1
        # profile = builder.create_optimization_profile()
        # profile.set_shape(input_tensor.name, tuple(shape), tuple(shape), tuple(shape))
        # For SCRFD
        input_tensor = network.get_input(0)
        static_shape  = tuple(input_tensor.shape)  # e.g. (32,3,640,640)
        profile = builder.create_optimization_profile()
        profile.set_shape(
            input_tensor.name,
            static_shape,   # min = static
            static_shape,   # opt = static
            static_shape    # max = static
        )

        config.add_optimization_profile(profile)

        # Detect TensorRT 10+ environment: no config.max_workspace_size attribute
        trt10 = not hasattr(config, 'max_workspace_size')
        if trt10:
            print("[INFO] Building serialized network for TensorRT 10+")
            serialized_engine = builder.build_serialized_network(network, config)
            engine_bytes = serialized_engine
        else:
            print("[INFO] Building engine for older TensorRT versions")
            engine = builder.build_engine(network, config)
            if engine is None:
                print("[ERROR] Engine build failed.")
                sys.exit(1)
            engine_bytes = engine.serialize()

        # Write engine to disk
        with open(engine_path, 'wb') as f:
            f.write(engine_bytes)
        print(f"[SUCCESS] Saved TensorRT engine to {engine_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX to TensorRT engine for TRT10+")
    parser.add_argument('--onnx', required=True, help='Path to ONNX model file')
    parser.add_argument('--engine', required=True, help='Path to save .engine file')
    parser.add_argument('--fp16', action='store_true', help='Enable FP16 precision')
    parser.add_argument('--batch_size', type=int, default=1, help='Max batch size (default:1)')
    parser.add_argument('--workspace', type=int, default=1024, help='Max workspace size in MB')
    args = parser.parse_args()

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        fp16=args.fp16,
        max_batch_size=args.batch_size,
        max_workspace_size=args.workspace
    )


if __name__ == '__main__':
    main()
