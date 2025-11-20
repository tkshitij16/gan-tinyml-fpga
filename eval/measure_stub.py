import time, statistics
import onnxruntime as ort
import numpy as np
from pathlib import Path

# Resolve model path relative to project root
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "compile" / "student_stub_simplified.onnx"
print("Measuring model:", MODEL_PATH)

def main():
    sess = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    x = np.random.rand(1, 1, 128, 128).astype("float32")
    times = []

    for _ in range(50):
        t0 = time.perf_counter()
        sess.run(None, {"input": x})
        times.append((time.perf_counter() - t0) * 1000)

    print(f"p50 = {statistics.median(times):.2f} ms, "
          f"p90 = {np.percentile(times, 90):.2f} ms")

if __name__ == "__main__":
    main()
