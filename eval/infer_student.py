from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "golden"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class TinyStub(nn.Module):
    """Tiny edge detector used as a stand-in for the student GAN."""

    def __init__(self):
        super().__init__()
        # Simple 3x3 Laplacian / edge detector kernel
        kernel = torch.tensor(
            [[-1.0, -1.0, -1.0],
             [-1.0,  8.0, -1.0],
             [-1.0, -1.0, -1.0]],
            dtype=torch.float32,
        )
        self.register_buffer("kernel", kernel.view(1, 1, 3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, H, W] in [0,1]
        y = F.conv2d(x, self.kernel, padding=1)
        y = torch.sigmoid(y)  # clamp to [0,1]
        return y


class StudentGenerator(nn.Module):
    """
    Wrapper for the student model.

    For now this just wraps TinyStub. Later this will load the
    compressed GAN student from gan-compression.
    """

    def __init__(self, ckpt_path: str | None = None):
        super().__init__()
        self.net = TinyStub()
        # TODO: when real student is ready, load weights here
        # if ckpt_path is not None:
        #     state = torch.load(ckpt_path, map_location="cpu")
        #     self.net.load_state_dict(state)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def save_golden_io(model: nn.Module, device: str = "cpu") -> None:
    model = model.to(device).eval()
    torch.manual_seed(0)

    # Simple synthetic "edge-like" pattern as input
    x = torch.zeros(1, 1, 128, 128, device=device)
    x[:, :, 32:96, 60:68] = 1.0  # a vertical bar

    with torch.no_grad():
        y = model(x)

    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()

    np.savez(LOG_DIR / "golden_io.npz", inp=x_np, out=y_np)
    print(f"Saved golden I/O to {LOG_DIR}")


def main():
    model = StudentGenerator()
    save_golden_io(model)


if __name__ == "__main__":
    main()
