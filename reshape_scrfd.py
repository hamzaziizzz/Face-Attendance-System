# reshape_scrfd.py  ▸  make batch static OR dynamic
import onnx
import argparse

def reshape(model, batch, h, w, dynamic=False):
    """
    If dynamic is True **or** batch < 0 → batch dim becomes symbolic.
    Otherwise the batch dimension is set to the positive integer `batch`.
    """
    def _set_dim(dim, value, symbolic=False):
        dim.ClearField("dim_value")
        dim.ClearField("dim_param")
        if symbolic:
            dim.dim_param = "batch_size"      # any name is fine
        else:
            dim.dim_value = value

    # ── INPUT ───────────────────────────────────────────────
    inp_dims = model.graph.input[0].type.tensor_type.shape.dim
    if dynamic or batch < 0:
        _set_dim(inp_dims[0], None, symbolic=True)
    else:
        _set_dim(inp_dims[0], batch)

    inp_dims[2].dim_value = h
    inp_dims[3].dim_value = w

    # ── OUTPUTS (make first dim match input) ───────────────
    for out in model.graph.output:
        out_dim = out.type.tensor_type.shape.dim[0]
        if dynamic or batch < 0:
            _set_dim(out_dim, None, symbolic=True)
        else:
            _set_dim(out_dim, batch)

    return model


def parse_args():
    p = argparse.ArgumentParser(
        description="Reshape SCRFD ONNX, optionally making batch dimension dynamic")
    p.add_argument("--onnx", required=True, help="Input ONNX path")
    p.add_argument("--out",  required=True, help="Output ONNX path")
    p.add_argument("--batch",  type=int, default=-1,
                   help="Batch size. Use -1 for dynamic (default: -1)")
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--width",  type=int, default=640)
    p.add_argument("--dynamic-batch", action="store_true",
                   help="Force dynamic batch even if --batch is positive")
    return p.parse_args()


def main():
    args = parse_args()

    model = onnx.load(args.onnx)
    reshaped = reshape(
        model,
        batch=args.batch,
        h=args.height,
        w=args.width,
        dynamic=args.dynamic_batch or args.batch < 0
    )
    onnx.save(reshaped, args.out)

    mode = "dynamic" if (args.dynamic_batch or args.batch < 0) else "static"
    bsz  = "symbolic" if mode == "dynamic" else str(args.batch)
    print(f"[SUCCESS] Saved {mode} ONNX → {args.out}  (batch={bsz}, size={args.width}×{args.height})")


if __name__ == "__main__":
    main()
