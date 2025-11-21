#!/usr/bin/env python3
"""
Evaluate StudentUNet against teacher (compressed GAN) outputs.

Metrics:
  - L1 loss (mean absolute error)
  - MSE (mean squared error)
  - PSNR (peak signal-to-noise ratio, in dB)
"""

import os
import sys
import math

from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ---------------------------
# Fix Python path to import from repo root
# ---------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from eval.student_unet import StudentUNet  # noqa: E402


class EdgeShoesStudentDataset(Dataset):
    """
    Loads paired images from:
      root/A/*.png  (edges)
      root/B/*.png  (teacher outputs)

    Assumes filenames match between A and B, e.g. 0_AB.png in both.
    """
    def __init__(self, root: str, image_size: int = 128):
        super().__init__()
        self.a_dir = os.path.join(root, "A")
        self.b_dir = os.path.join(root, "B")
        if not os.path.isdir(self.a_dir) or not os.path.isdir(self.b_dir):
            raise RuntimeError(f"Expected A/ and B/ under {root}")

        self.ids = sorted(
            f for f in os.listdir(self.a_dir)
            if f.lower().endswith(".png")
        )
        if not self.ids:
            raise RuntimeError(f"No PNG files found in {self.a_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        fname = self.ids[idx]
        a_path = os.path.join(self.a_dir, fname)
        b_path = os.path.join(self.b_dir, fname)

        edge = Image.open(a_path).convert("RGB")
        shoe = Image.open(b_path).convert("RGB")

        edge = self.transform(edge)   # [0,1]
        shoe = self.transform(shoe)   # [0,1]

        return edge, shoe, fname


def evaluate(
    data_root: str = "data/edges2shoes_student",
    checkpoint_path: str = "checkpoints/student_unet.pth",
    image_size: int = 128,
    base_ch: int = 16,
    batch_size: int = 1,
    num_workers: int = 2,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset & loader
    dataset = EdgeShoesStudentDataset(data_root, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    print(f"Dataset size: {len(dataset)} samples")

    # Load model
    ckpt_path = os.path.join(ROOT, checkpoint_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = StudentUNet(in_ch=3, out_ch=3, base_ch=base_ch).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    total_l1 = 0.0
    total_mse = 0.0
    total_psnr = 0.0
    count = 0

    with torch.no_grad():
        for edges, shoes, fnames in loader:
            edges = edges.to(device)   # B,3,H,W
            shoes = shoes.to(device)   # B,3,H,W

            preds = model(edges)       # B,3,H,W

            # Make sure range is [0,1]
            preds = torch.clamp(preds, 0.0, 1.0)

            # Compute metrics per-batch
            l1 = F.l1_loss(preds, shoes, reduction="mean").item()
            mse = F.mse_loss(preds, shoes, reduction="mean").item()

            if mse > 0:
                psnr = 10.0 * math.log10(1.0 / mse)  # max=1.0
            else:
                psnr = float("inf")

            total_l1 += l1
            total_mse += mse
            total_psnr += psnr
            count += 1

    avg_l1 = total_l1 / count
    avg_mse = total_mse / count
    avg_psnr = total_psnr / count

    print("\n==== Student vs Teacher Metrics (on dataset) ====")
    print(f"Samples: {count}")
    print(f"Avg L1   : {avg_l1:.6f}")
    print(f"Avg MSE  : {avg_mse:.6f}")
    print(f"Avg PSNR : {avg_psnr:.2f} dB")
    print("===============================================\n")


def main():
    data_root = os.path.join(ROOT, "data", "edges2shoes_student")
    checkpoint_path = os.path.join("checkpoints", "student_unet.pth")
    evaluate(
        data_root=data_root,
        checkpoint_path=checkpoint_path,
        image_size=128,
        base_ch=16,
        batch_size=1,
        num_workers=2,
    )


if __name__ == "__main__":
    main()
