import os, sys
from pathlib import Path

# Make project root importable: <root>/eval, <root>/compile, etc.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import onnx
import onnxsim
from eval.infer_student import TinyStub   # or StudentGenerator later

def main():
    model = TinyStub().eval()
    dummy = torch.randn(1, 1, 128, 128)

    out_path = "compile/student_stub.onnx"

    # Export at opset 18 and DON'T request dynamic_axes (for now)
    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=18,          # <- use a modern, supported opset
        do_constant_folding=True,  # fold constants to simplify graph
        dynamic_axes=None,         # keep it simple; fixed batch/shape for now
    )

    print("Exported", out_path)

    # Load and simplify with onnxsim
    model_onnx = onnx.load(out_path)
    model_simp, check = onnxsim.simplify(
        model_onnx,
        dynamic_input_shape=False,
    )
    assert check, "ONNX simplify failed"

    simp_path = "compile/student_stub_simplified.onnx"
    onnx.save(model_simp, simp_path)
    print("Saved simplified model to", simp_path)

if __name__ == "__main__":
    main()

