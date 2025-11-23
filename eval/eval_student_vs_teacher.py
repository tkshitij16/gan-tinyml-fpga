#!/usr/bin/env python3
import os
import argparse
import glob
import numpy as np
import onnxruntime as ort
from PIL import Image

# ---------------------------------------------------------------------
# Helpers to load edge (A) and teacher (B) images
# ---------------------------------------------------------------------

def load_img(path, size=(256, 256)):
    """
    Load edge image (grayscale) as 1x1xHxW float32 in [0,1].
    """
    img = Image.open(path).convert("L")
    img = img.resize(size, Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    # shape: (H, W) -> (1, 1, H, W)
    arr = arr[None, None, :, :]
    return arr


def load_teacher_img(path, size=(256, 256)):
    """
    Load teacher image (RGB) as 1x3xHxW float32 in [0,1].
    """
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    # shape: (H, W, C) -> (1, C, H, W)
    arr = arr.transpose(2, 0, 1)[None, :, :, :]
    return arr


def psnr(mse: float) -> float:
    if mse == 0:
        return 100.0
    return 20.0 * np.log10(1.0 / np.sqrt(mse))


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def eval_student(data_root: str, onnx_path: str, limit: int | None = None):
    print(f"[Eval] data_root = {data_root}")
    print(f"[Eval] onnx_path = {onnx_path}")

    # Our dataset is in:
    #   data_root/A  -> edges
    #   data_root/B  -> teacher outputs
    edge_dir = os.path.join(data_root, "A")
    teacher_dir = os.path.join(data_root, "B")

    if not (os.path.isdir(edge_dir) and os.path.isdir(teacher_dir)):
        raise FileNotFoundError(f"Expected {edge_dir} and {teacher_dir} to exist")

    edge_files = sorted(glob.glob(os.path.join(edge_dir, "*.png")))
    teacher_files = sorted(glob.glob(os.path.join(teacher_dir, "*.png")))

    if not edge_files or not teacher_files:
        raise RuntimeError(f"No PNG files found in {edge_dir} or {teacher_dir}")

    if limit is not None:
        edge_files = edge_files[:limit]
        teacher_files = teacher_files[:limit]

    print(f"[Eval] Number of pairs: {min(len(edge_files), len(teacher_files))}")

    # ONNX Runtime session (CPU)
    sess = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )

    L1_list = []
    MSE_list = []
    PSNR_list = []

    for e_path, t_path in zip(edge_files, teacher_files):
        x = load_img(e_path, size=(256, 256))
        t = load_teacher_img(t_path, size=(256, 256))

        # Run ONNX model
        out = sess.run(["output"], {"input": x})[0]
        out = np.clip(out, 0.0, 1.0)

        diff = out - t
        l1 = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff ** 2))
        ps = psnr(mse)

        L1_list.append(l1)
        MSE_list.append(mse)
        PSNR_list.append(ps)

    print("==== Student vs Teacher Metrics (on dataset) ====")
    print(f"Samples: {len(L1_list)}")
    print(f"Avg L1   : {np.mean(L1_list):.6f}")
    print(f"Avg MSE  : {np.mean(MSE_list):.6f}")
    print(f"Avg PSNR : {np.mean(PSNR_list):.2f} dB")
    print("===============================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--onnx-path", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    eval_student(args.data_root, args.onnx_path, limit=args.limit)


if __name__ == "__main__":
    main()
