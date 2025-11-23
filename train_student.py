#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------------------------------------------------------
# Ensure project root is on sys.path so we can import eval.student_unet
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from eval.student_unet import StudentUNet
except Exception as e:
    print("ERROR: Could not import StudentUNet from eval.student_unet")
    print("Exception:", repr(e))
    raise

IMG_SIZE = 256  # must match ONNX + UI


class EdgeToShoeDataset(Dataset):
    """
    Loads paired (edge, teacher) images from:

        data_root/A   -> edge / sketch
        data_root/B   -> teacher / fake_B

    This matches tools/make_student_dataset.py which writes A/ and B/.
    """

    def __init__(self, data_root: str, img_size: int = IMG_SIZE):
        self.data_root = Path(data_root)
        self.edge_dir = self.data_root / "A"
        self.teacher_dir = self.data_root / "B"
        self.img_size = img_size

        print(f"[Dataset] data_root = {self.data_root}")
        print(f"[Dataset] edge_dir  = {self.edge_dir}")
        print(f"[Dataset] teacher_dir = {self.teacher_dir}")

        if not self.edge_dir.is_dir() or not self.teacher_dir.is_dir():
            raise FileNotFoundError(
                f"Expected {self.edge_dir} and {self.teacher_dir} to exist.\n"
                f"Hint: run tools/make_student_dataset.py first."
            )

        edge_files = sorted(
            f for f in os.listdir(self.edge_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        teacher_files = sorted(
            f for f in os.listdir(self.teacher_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )

        n = min(len(edge_files), len(teacher_files))
        self.edge_files = edge_files[:n]
        self.teacher_files = teacher_files[:n]

        if n == 0:
            raise RuntimeError(f"No image pairs found under {self.data_root}")

        print(f"Dataset size: {n} samples")

    def __len__(self):
        return len(self.edge_files)

    def _load_edge(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("L")  # grayscale edges
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # [H,W] in [0,1]
        return arr[None, :, :]  # (1, H, W)

    def _load_teacher(self, path: Path) -> np.ndarray:
        img = Image.open(path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0  # [H,W,3]
        return arr.transpose(2, 0, 1)  # (3, H, W)

    def __getitem__(self, idx):
        edge_name = self.edge_files[idx]
        teacher_name = self.teacher_files[idx]

        edge_path = self.edge_dir / edge_name
        teacher_path = self.teacher_dir / teacher_name

        edge = self._load_edge(edge_path)
        teacher = self._load_teacher(teacher_path)

        edge_tensor = torch.from_numpy(edge)        # (1,H,W)
        teacher_tensor = torch.from_numpy(teacher)  # (3,H,W)
        return edge_tensor, teacher_tensor


def train(
    data_root: str,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
    checkpoint_path: str = "checkpoints/student_unet.pth",
):
    print("[Train] Starting training loop...")
    print(f"[Train] data_root      = {data_root}")
    print(f"[Train] epochs         = {epochs}")
    print(f"[Train] batch_size     = {batch_size}")
    print(f"[Train] lr             = {lr}")
    print(f"[Train] device         = {device}")
    print(f"[Train] checkpoint     = {checkpoint_path}")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    dataset = EdgeToShoeDataset(data_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )

    print("[Train] Building StudentUNet...")
    model = StudentUNet(in_channels=1, out_channels=3, base_channels=32)
    model.to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")

    print(f"[Train] Using device: {device}")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        pbar = tqdm(enumerate(loader), total=len(loader), ncols=100)
        for step, (edges, teachers) in pbar:
            edges = edges.to(device)
            teachers = teachers.to(device)

            optimizer.zero_grad()
            outputs = model(edges)
            loss = criterion(outputs, teachers)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * edges.size(0)

            if step % 20 == 0:
                pbar.set_description(
                    f"Epoch {epoch}/{epochs} Step {step}/{len(loader)} "
                    f"Loss {loss.item():.4f}"
                )

        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch} average L1 loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "avg_loss": avg_loss,
                    "img_size": IMG_SIZE,
                },
                checkpoint_path,
            )
            print(f"Saved new best model to {checkpoint_path}")

    print(f"Training finished. Best avg loss: {best_loss:.4f}")


def main():
    print("[Main] train_student.py starting...")
    parser = argparse.ArgumentParser(
        description="Train StudentUNet on edges2shoes student dataset (A→B)."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/edges2shoes_student",
        help="Root folder containing A/ and B/ subfolders",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/student_unet.pth",
    )
    args = parser.parse_args()
    print("[Main] Parsed args:", args)

    try:
        train(
            data_root=args.data_root,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            checkpoint_path=args.checkpoint,
        )
    except Exception as e:
        print("[Main] ERROR during training:", repr(e))
        raise


if __name__ == "__main__":
    main()
