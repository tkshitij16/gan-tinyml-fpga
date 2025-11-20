import gradio as gr
import numpy as np
import onnxruntime as ort
import cv2
from pathlib import Path

# Resolve model path relative to this file (ui/app_stub.py)
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "compile" / "student_stub_simplified.onnx"

print("Loading model from:", MODEL_PATH)

sess = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

def run(image):
    # Convert to grayscale
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize to model input shape
    img = cv2.resize(img, (128, 128))
    # Normalize
    x = img.astype("float32") / 255.0
    x = x[None, None, :, :]  # shape: [1, 1, 128, 128]

    # Run through ONNX model
    y = sess.run(None, {"input": x})[0][0, 0]

    # Convert back to uint8 image
    y = (np.clip(y, 0, 1) * 255).astype("uint8")
    return y

demo = gr.Interface(
    fn=run,
    inputs="image",
    outputs="image",
    title="Edges → Shoes (Stub Generator)",
    description="This is a placeholder model. Replace with real compressed GAN later."
)

if __name__ == "__main__":
    demo.launch()

