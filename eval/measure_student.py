# File: eval/measure_student.py
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

# -------------------------------------------------------------------
# Ensure project root on sys.path (not strictly needed here, but safe)
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMG_SIZE = 256


def measure_latency(onnx_path: Path, runs: int = 100):
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

    print(f"Loading ONNX model from: {onnx_path}")
    sess = ort.InferenceSession(
        onnx_path.as_posix(),
        providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Dummy input in [0,1]
    dummy = np.random.rand(1, 1, IMG_SIZE, IMG_SIZE).astype(np.float32)

    # Warmup
    for _ in range(10):
        sess.run([output_name], {input_name: dummy})

    times = []
    for i in range(runs):
        t0 = time.time()
        sess.run([output_name], {input_name: dummy})
        t1 = time.time()
        times.append((t1 - t0) * 1000.0)  # ms

    times = np.array(times, dtype=np.float64)
    p50 = np.percentile(times, 50)
    p90 = np.percentile(times, 90)
    mean = times.mean()
    std = times.std()

    print("Latency over {} runs:".format(runs))
    print(f"  p50  = {p50:.2f} ms")
    print(f"  p90  = {p90:.2f} ms")
    print(f"  mean = {mean:.2f} ms")
    print(f"  std  = {std:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description="Measure StudentUNet latency via ONNXRuntime")
    parser.add_argument(
        "--onnx-path",
        type=str,
        default="compile/student_unet_simplified.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    onnx_path = (ROOT / args.onnx_path).resolve()
    measure_latency(onnx_path, runs=args.runs)


if __name__ == "__main__":
    main()
