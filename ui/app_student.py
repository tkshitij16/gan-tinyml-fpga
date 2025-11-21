#!/usr/bin/env python3
import sys
import os

# Add project root to PYTHONPATH
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

import time
import gradio as gr
import numpy as np
import onnxruntime as ort
from PIL import Image


def load_session():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx_path = os.path.join(repo_root, "compile", "student_unet_simplified.onnx")
    if not os.path.exists(onnx_path):
        raise SystemExit(f"ONNX model not found: {onnx_path}")
    print(f"Loading ONNX model from: {onnx_path}")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return sess


sess = load_session()
input_name = sess.get_inputs()[0].name
output_name = sess.get_outputs()[0].name


def preprocess(x: np.ndarray, size=(128, 128)) -> np.ndarray:
    """
    x: H x W x 3, uint8 or float image from Gradio.
    """
    if x.dtype != np.uint8:
        x = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(x).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # [0,1]
    arr = np.transpose(arr, (2, 0, 1))  # C,H,W
    arr = arr[None, ...]  # 1,C,H,W
    return arr


def postprocess(y: np.ndarray) -> np.ndarray:
    """
    y: 1 x C x H x W -> H x W x C, uint8
    """
    y = y[0]
    y = np.transpose(y, (1, 2, 0))
    y = np.clip(y, 0.0, 1.0)
    y = (y * 255.0).astype(np.uint8)
    return y


def run_student(edge_np: np.ndarray) -> np.ndarray:
    if edge_np is None:
        return None
    inp = preprocess(edge_np, size=(128, 128))
    out = sess.run([output_name], {input_name: inp})[0]
    return postprocess(out)


with gr.Blocks() as demo:
    gr.Markdown("# StudentUNet Demo (Edges → Shoes)")

    with gr.Row():
        in_img = gr.Image(type="numpy", label="Edge input")
        out_img = gr.Image(type="numpy", label="Student output")

    btn = gr.Button("Generate")
    btn.click(fn=run_student, inputs=in_img, outputs=out_img)

demo.launch()
