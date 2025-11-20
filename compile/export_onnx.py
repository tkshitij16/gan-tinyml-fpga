import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import onnx
import onnxsim
from eval.infer_student import StudentGenerator


def main():
    model = StudentGenerator().eval()
    dummy = torch.randn(1, 1, 128, 128)

    out_path = ROOT / "compile" / "student_stub.onnx"
    torch.onnx.export(
        model,
        dummy,
        out_path.as_posix(),
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        do_constant_folding=True,
        dynamic_axes=None,
    )
    print("Exported", out_path)

    model_onnx = onnx.load(out_path.as_posix())
    model_simp, check = onnxsim.simplify(model_onnx, dynamic_input_shape=False)
    assert check, "ONNX simplify failed"
    simp_path = ROOT / "compile" / "student_stub_simplified.onnx"
    onnx.save(model_simp, simp_path.as_posix())
    print("Saved simplified model to", simp_path)


if __name__ == "__main__":
    main()
