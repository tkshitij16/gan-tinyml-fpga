# eval/infer_student.py

from pathlib import Path
import numpy as np
from PIL import Image
import torch

from eval.student_unet import TinyStub


def load_edge_image(path: Path, size: int = 128) -> torch.Tensor:
    """
    Load an edge image and convert to normalized tensor in [0, 1], shape (1, 1, H, W).
    """
    img = Image.open(path).convert("L")  # grayscale
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img).astype("float32") / 255.0
    arr = arr[None, None, ...]  # NCHW
    return torch.from_numpy(arr)


def tensor_to_rgb_image(t: torch.Tensor) -> Image.Image:
    """
    Convert output tensor in [-1, 1] with shape (1, 3, H, W) to PIL RGB image.
    """
    t = t.detach().cpu().clamp(-1, 1)
    t = (t + 1.0) / 2.0  # -> [0, 1]
    t = (t * 255.0).numpy()
    t = t[0].transpose(1, 2, 0).astype("uint8")
    return Image.fromarray(t)


def main():
    root = Path(__file__).resolve().parents[1]
    logs_dir = root / "logs" / "golden"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # model
    model = TinyStub()
    ckpt_path = root / "checkpoints" / "student_unet.pth"
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print("WARNING: No checkpoint found, using randomly initialized StudentUNet.")

    model.eval()

    # input
    sample_edge = root / "assets" / "sample_edge.png"
    if sample_edge.exists():
        x = load_edge_image(sample_edge, size=128)
        print(f"Using edge image: {sample_edge}")
    else:
        # fallback: simple synthetic square
        x = torch.zeros(1, 1, 128, 128)
        x[:, :, 32:96, 32:96] = 1.0
        print("No assets/sample_edge.png found. Using synthetic square pattern.")

    # forward
    with torch.no_grad():
        y = model(x)

    # save golden I/O for ONNX checks
    npz_path = logs_dir / "golden_io.npz"
    np.savez_compressed(
        npz_path,
        input=x.numpy().astype("float32"),
        output=y.numpy().astype("float32"),
    )

    out_img = tensor_to_rgb_image(y)
    out_path = logs_dir / "out_student.png"
    out_img.save(out_path)

    print("Saved golden I/O to", npz_path)
    print("Saved preview image to", out_path)


if __name__ == "__main__":
    main()
