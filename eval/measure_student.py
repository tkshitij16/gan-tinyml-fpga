#!/usr/bin/env python3
"""
Measure latency of student_unet_simplified.onnx on CPU.
"""

import os
import sys
import time
import numpy as np
import onnxruntime as ort

# ---------------------------
# Fix Python PATH for imports
# ---------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

COMPILE_DIR = os.path.join(ROOT, "compile")
ONNX_PATH = os.path.join(COMPILE_DIR, "student_unet_simplified.onnx")


def main():
    # ---------------------------
    # Validate ONNX file exists
    # ---------------------------
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(
            f"ONNX model not found at {ONNX_PATH}\n"
            f"Run: python compile/export_onnx.py"
        )

    print(f"Loading ONNX model from: {ONNX_PATH}")

    # ---------------------------
    # Create ORT inference session (CPU only)
    # ---------------------------
    providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(
        ONNX_PATH,
        providers=providers
    )

    input_name = sess.get_inputs()[0].name

    # ---------------------------
    # Prepare dummy input for timing
    # ---------------------------
    dummy = np.random.rand(1, 3, 128, 128).astype(np.float32)

    # Warm-up
    for _ in range(5):
        sess.run(None, {input_name: dummy})

    # ---------------------------
    # Timing loop
    # ---------------------------
    times = []
    N = 100  # number of runs

    print(f"Running {N} inference loops...")

    for _ in range(N):
        t0 = time.time()
        sess.run(None, {input_name: dummy})
        times.append((time.time() - t0) * 1000.0)  # ms

    times = np.array(times)

    # ---------------------------
    # Print results
    # ---------------------------
    print("\nLatency over {} runs:".format(N))
    print(f"  p50  = {np.percentile(times, 50):.2f} ms")
    print(f"  p90  = {np.percentile(times, 90):.2f} ms")
    print(f"  mean = {np.mean(times):.2f} ms")
    print(f"  std  = {np.std(times):.2f} ms\n")


if __name__ == "__main__":
    main()
