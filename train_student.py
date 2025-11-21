#!/usr/bin/env python3
import argparse
import os

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from eval.student_unet import StudentUNet


class EdgeShoesStudentDataset(Dataset):
    """
    Loads paired images from:
      root/A/*.png  (edges)
      root/B/*.png  (teacher outputs)

    Assumes filenames match between A and B, e.g.:
      A/0_AB.png  and  B/0_AB.png
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

        edge = self.transform(edge)
        shoe = self.transform(shoe)

        # Both in [0,1], L1 loss will work fine
        return edge, shoe


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = EdgeShoesStudentDataset(args.data_root, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    model = StudentUNet(in_ch=3, out_ch=3, base_ch=args.base_ch).to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_idx, (edge, shoe) in enumerate(loader):
            edge = edge.to(device)
            shoe = shoe.to(device)

            optimizer.zero_grad()
            pred = model(edge)
            loss = criterion(pred, shoe)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 20 == 0:
                print(
                    f"Epoch {epoch}/{args.epochs} "
                    f"Step {batch_idx}/{len(loader)} "
                    f"Loss {loss.item():.4f}"
                )

        avg_loss = running_loss / len(loader)
        print(f"Epoch {epoch} average L1 loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = os.path.join(args.checkpoint_dir, "student_unet.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved new best model to {ckpt_path}")

    print(f"Training finished. Best avg loss: {best_loss:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train StudentUNet on edges2shoes student dataset."
    )
    parser.add_argument("--data-root", type=str, default="data/edges2shoes_student")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-ch", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
