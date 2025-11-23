# File: compile/export_onnx.py
import os
import sys
from pathlib import Path

import torch
import onnx
import onnxsim

# -------------------------------------------------------------------
# Ensure project root is on sys.path so `import eval.student_unet` works
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.student_unet import StudentUNet  # noqa: E402

IMG_SIZE = 256  # must match training / eval / UI


def main():
    root = ROOT
    ckpt_path = root / "checkpoints" / "student_unet.pth"
    out_dir = root / "compile"
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = out_dir / "student_unet.onnx"
    simp_onnx_path = out_dir / "student_unet_simplified.onnx"

    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. Train the student first."
        )

    # Build model
    model = StudentUNet(in_channels=1, out_channels=3, base_channels=32)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE, dtype=torch.float32)

    print(f"Exporting ONNX to {onnx_path} ...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path.as_posix(),
        input_names=["input"],
        output_names=["output"],
        opset_version=18,          # modern opset, avoids version-conversion issues
        dynamic_axes=None,         # fixed batch=1, resolution=256x256
    )
    print(f"Exported ONNX to {onnx_path}")

    # Simplify
    print("Simplifying ONNX model ...")
    model_onnx = onnx.load(onnx_path.as_posix())
    model_simp, check = onnxsim.simplify(model_onnx)
    if not check:
        raise RuntimeError("onnx-simplifier check failed")
    onnx.save(model_simp, simp_onnx_path.as_posix())
    print(f"Saved simplified ONNX to {simp_onnx_path}")


if __name__ == "__main__":
    main()
