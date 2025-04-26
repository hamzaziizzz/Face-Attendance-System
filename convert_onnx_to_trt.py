#!/usr/bin/env python3
"""
Build a TensorRT engine with **dynamic batch support**.
Works for TensorRT>=8.x and 10.x (serialized network path).
"""

import os
import sys
import argparse
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
EXPL_BATCH  = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def build_engine(onnx_path: str,
                 engine_path: str,
                 fp16: bool,
                 min_batch: int,
                 opt_batch: int,
                 max_batch: int,
                 max_workspace_mb: int = 1024):

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPL_BATCH) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        # ── Workspace & precision ―――――――――――――――――――――
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            max_workspace_mb * (1 << 20)
        )
        if fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # ── Parse ONNX ――――――――――――――――――――――――――――――――
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                sys.exit("[ERROR] ONNX parsing failed")

        # ── Build optimisation profile for dynamic batch ―
        profile = builder.create_optimization_profile()
        for i in range(network.num_inputs):
            t = network.get_input(i)
            spatial = tuple(t.shape[1:])        # (C,H,W)
            profile.set_shape(
                t.name,
                (min_batch, *spatial),          # min
                (opt_batch, *spatial),          # opt
                (max_batch, *spatial)           # max
            )
        config.add_optimization_profile(profile)

        # ── Build / serialize ――――――――――――――――――――――――――
        if not hasattr(config, "max_workspace_size"):    # TRT ≥10
            print("[INFO] TRT10+: serialising network …")
            engine_bytes = builder.build_serialized_network(network, config)
        else:                                            # TRT 8/9
            print("[INFO] TRT ≤9: building engine …")
            engine = builder.build_engine(network, config)
            if engine is None:
                sys.exit("[ERROR] Engine build failed")
            engine_bytes = engine.serialize()

        with open(engine_path, "wb") as f:
            f.write(engine_bytes)
        print(f"[SUCCESS] Engine saved ➜ {engine_path}")

def cli():
    ap = argparse.ArgumentParser(
        description="Convert ONNX ➜ TensorRT with dynamic batch")
    ap.add_argument("--onnx",   required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--fp16",   action="store_true")
    ap.add_argument("--min_batch", type=int, default=1,
                    help="minimum batch for profile (default 1)")
    ap.add_argument("--opt_batch", type=int, default=64,
                    help="optimal batch for profile (default 64)")
    ap.add_argument("--max_batch", type=int, default=64,
                    help="maximum batch for profile (default 64)")
    ap.add_argument("--workspace", type=int, default=1024,
                    help="workspace size in MB (default 1024)")
    args = ap.parse_args()

    if not (1 <= args.min_batch <= args.opt_batch <= args.max_batch):
        sys.exit("[ERROR] min ≤ opt ≤ max must hold")

    build_engine(args.onnx, args.engine, args.fp16,
                 args.min_batch, args.opt_batch, args.max_batch,
                 args.workspace)

if __name__ == "__main__":
    cli()