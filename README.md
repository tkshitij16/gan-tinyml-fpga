# GAN Compression → FPGA TinyML Project

Root layout:
- gan-compression/   # upstream MIT repo
- compress/          # our scripts to train/compress student models
- quantize/          # Vitis AI / QAT scripts
- compile/           # ONNX export + FPGA compilation
- deploy/            # board-side runtime code
- eval/              # metrics: latency, FID/LPIPS, etc.
- ui/                # Gradio / UI code
- calib/             # calibration image lists
- logs/              # logs, golden I/O, etc.
