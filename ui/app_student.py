#!/usr/bin/env python3
import os
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image
import onnxruntime as ort


# -------------------------------------------------------------
# Model loading
# -------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
MODEL_PATH = ROOT_DIR / "compile" / "student_unet_simplified.onnx"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ONNX model not found at {MODEL_PATH}. "
        "Run `python compile/export_onnx.py` first."
    )

print(f"Loading ONNX model from: {MODEL_PATH}")
sess = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------
def _extract_image_from_editor(value):
    """
    Accepts Gradio's ImageEditor / Image value and returns a NumPy array or PIL.Image.
    Supported dict shapes (vary by gradio version):
      - {'composite': np.ndarray, 'background': np.ndarray, 'layers': [...]}
      - {'image': np.ndarray, 'mask': np.ndarray}
      - {'background': np.ndarray, ...}
    Falls back to returning the value if it's already a PIL.Image or np.ndarray.
    """
    if value is None:
        raise gr.Error("Please upload or draw an edge image first.")

    if isinstance(value, dict):
        # Try keys in a safe order without using boolean `or` on arrays.
        for key in ("composite", "image", "background"):
            if key in value and value[key] is not None:
                return value[key]
        raise gr.Error("Could not extract an image from the editor input.")
    return value


def preprocess_edge(edge_value, desired_size=(256, 256)):
    """
    Convert editor/image input to NCHW float32 in [0,1], grayscale (1 channel).
    """
    img = _extract_image_from_editor(edge_value)

    # To PIL
    if isinstance(img, np.ndarray):
        pil = Image.fromarray(img)
    elif isinstance(img, Image.Image):
        pil = img
    else:
        # If it's a path-like
        pil = Image.open(img)

    # Grayscale + resize to model's expected spatial size
    pil = pil.convert("L").resize(desired_size, Image.BILINEAR)

    arr = np.array(pil).astype("float32") / 255.0  # HxW
    arr = arr[None, None, :, :]  # 1x1xHxW
    return arr


def run_inference(edge_value):
    """
    Gradio callback: take edited edge image -> ONNX -> RGB output PIL.Image
    """
    # Infer model's spatial size from input metadata (fallback to 256x256)
    in_meta = sess.get_inputs()[0]
    # Shapes can be [None, 1, 256, 256] or [1, 1, 256, 256]. Use last two dims if present.
    h = (in_meta.shape[2] if len(in_meta.shape) > 2 and isinstance(in_meta.shape[2], int) else 256)
    w = (in_meta.shape[3] if len(in_meta.shape) > 3 and isinstance(in_meta.shape[3], int) else 256)

    x = preprocess_edge(edge_value, desired_size=(w, h))  # NCHW float32

    # ONNX forward
    y = sess.run(["output"], {"input": x})[0]  # 1x3xHxW

    if y.ndim != 4 or y.shape[0] != 1 or y.shape[1] != 3:
        raise gr.Error(f"Unexpected model output shape: {y.shape}")

    y = np.clip(y[0], 0.0, 1.0)  # 3xHxW
    y = (y * 255).astype("uint8")
    y = np.transpose(y, (1, 2, 0))  # HxWx3
    return Image.fromarray(y)


# -------------------------------------------------------------
# UI
# -------------------------------------------------------------
def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
# Edge2Shoes — Compressed Student GAN (Interactive)

**How to use**
1. Upload an *edge* image of a shoe **or** draw from scratch.
2. Use the brush/eraser to edit the edges.
3. Click **Generate shoe** to run the student model.
"""
        )

        with gr.Row():
            edge_editor = gr.ImageEditor(
                label="1) Upload or Draw Edges (use brush/eraser)",
                type="numpy",          # returns dict with numpy arrays
                interactive=True,
            )
            output_img = gr.Image(
                label="2) Generated Shoe (Student ONNX)",
                type="numpy"
            )

        run_btn = gr.Button("Generate shoe")

        run_btn.click(
            fn=run_inference,
            inputs=edge_editor,
            outputs=output_img,
        )

        gr.Markdown(
            """
**Tips**
- Use thin, continuous outlines for best results (like the edges2shoes dataset).
- Erase parts to see how the generator reacts in real time.
- This UI uses your exported **ONNX student** model.
"""
        )
    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch()
