# reshape_scrfd.py
import onnx, argparse

def reshape(model, n: int, h: int, w: int):
    inp = model.graph.input[0].type.tensor_type.shape.dim
    inp[0].dim_value = n      # batch size
    inp[2].dim_value = h
    inp[3].dim_value = w
    return model

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx',   required=True)
    p.add_argument('--out',    required=True)
    p.add_argument('--batch',  type=int, default=1)
    p.add_argument('--height', type=int, default=640)
    p.add_argument('--width',  type=int, default=640)
    args = p.parse_args()

    model = onnx.load(args.onnx)
    reshaped = reshape(model,
                       n=args.batch,
                       h=args.height,
                       w=args.width)
    onnx.save(reshaped, args.out)
    print(f"Saved static ONNX with batch={args.batch}, size={args.width}×{args.height}")

if __name__ == '__main__':
    main()
