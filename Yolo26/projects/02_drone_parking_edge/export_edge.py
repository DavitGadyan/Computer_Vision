"""Export YOLO26-OBB for edge devices and benchmark the formats that can run on this host's CPU.

Targets:
  * ONNX      - generic (ONNX Runtime on x86 / ARM)
  * NCNN      - Raspberry Pi 4/5 CPU (fastest ARM CPU backend)
  * OpenVINO  - Intel NUC / industrial PCs
  * IMX500    - Raspberry Pi AI Camera (on-sensor NPU)  -> `--imx` (needs Sony MCT toolchain, x86 Linux)
  * TensorRT  - NVIDIA Jetson Orin/Nano: run `yolo export model=yolo26n-obb.pt format=engine half=True` ON the Jetson
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from config import DEVICE, MODELS, model_path  # noqa: E402

from ultralytics import YOLO  # noqa: E402


def bench(weights, frames, imgsz, runs=30):
    m = YOLO(weights, task="obb")
    for f in frames[:3]:
        m(f, imgsz=imgsz, device=DEVICE, verbose=False)
    t, n = [], 0
    for k in range(runs):
        f = frames[k % len(frames)]
        s = time.perf_counter()
        r = m(f, imgsz=imgsz, device=DEVICE, verbose=False)[0]
        t.append(time.perf_counter() - s)
        n += len(r.obb)
    return 1000 * float(np.median(t)), n / runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26n-obb.pt")
    ap.add_argument("--source", default="data/parking_drone.mp4")
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 1024])
    ap.add_argument("--formats", nargs="+", default=["onnx", "ncnn", "openvino"])
    ap.add_argument("--imx", action="store_true", help="also export for Raspberry Pi AI Camera (IMX500, INT8)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.source)
    frames = []
    while len(frames) < 30:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (1280, 720)))
    pt = model_path(args.model)
    rows = []
    for sz in args.imgsz:
        ms, det = bench(pt, frames, sz)
        rows.append({"format": "pytorch", "imgsz": sz, "ms": round(ms, 1), "fps": round(1000 / ms, 1), "dets": round(det, 1)})
        for fmt in args.formats:
            exported = YOLO(pt).export(format=fmt, imgsz=sz)
            ms, det = bench(exported, frames, sz)
            rows.append({"format": fmt, "imgsz": sz, "ms": round(ms, 1), "fps": round(1000 / ms, 1), "dets": round(det, 1)})
            print(rows[-1])
    if args.imx:
        try:
            YOLO(pt).export(format="imx", imgsz=640, data="DOTAv1.yaml")
            rows.append({"format": "imx500", "imgsz": 640, "ms": None, "fps": None, "dets": None, "note": "exported; runs on-sensor"})
        except Exception as e:  # toolchain is heavy / x86-only; document rather than fail
            rows.append({"format": "imx500", "note": f"export failed on this host: {type(e).__name__}"})

    print("\n| Format | imgsz | latency (ms) | FPS | avg detections |\n|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['format']} | {r.get('imgsz', '')} | {r.get('ms', '')} | {r.get('fps', '')} | {r.get('dets', '')} |")
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/edge_benchmark.json").write_text(json.dumps(rows, indent=2))
    print(f"\nexports live next to the weights in {MODELS} (gitignored)")


if __name__ == "__main__":
    main()
