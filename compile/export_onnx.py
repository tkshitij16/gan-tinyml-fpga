#!/usr/bin/env python3
import sys
import os

# Fix import path so eval.student_unet works anywhere
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import onnx
import onnxsim

from eval.student_unet import StudentUNet


def main():
    repo_root = ROOT
    ckpt_path = os.path.join(repo_root, "checkpoints", "student_unet.pth")
    out_dir = os.path.join(repo_root, "compile")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    model = StudentUNet(in_ch=3, out_ch=3, base_ch=16)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 128, 128, dtype=torch.float32)
    onnx_path = os.path.join(out_dir, "student_unet.onnx")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        dynamic_axes={"input": [0], "output": [0]},
    )
    print(f"Exported ONNX to {onnx_path}")

    model_onnx = onnx.load(onnx_path)
    model_simp, check = onnxsim.simplify(model_onnx)
    if not check:
        raise SystemExit("ONNX simplification failed")

    simp_path = os.path.join(out_dir, "student_unet_simplified.onnx")
    onnx.save(model_simp, simp_path)
    print(f"Saved simplified ONNX to {simp_path}")


if __name__ == "__main__":
    main()
