"""Median CPU inference latency of every model used in the portfolio (PyTorch, batch 1).

Usage: python scripts/benchmark_cpu.py   ->  prints a markdown table, writes outputs/cpu_benchmark.json
"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from config import DEVICE, MODELS, ROOT  # noqa: E402

from ultralytics import YOLO, YOLOE  # noqa: E402

P = ROOT / "projects"
CASES = [
    ("01 Workout", "yolo26n-pose.pt", 640, P / "01_workout_monitoring/data/pushups.mp4"),
    ("02 Parking (OBB)", "yolo26n-obb.pt", 1024, P / "02_drone_parking_edge/data/parking_drone.mp4"),
    ("03 Solar defects (custom)", "solar_defect_yolo26n.pt", 640, None),
    ("04 Traffic (seg)", "yolo26n-seg.pt", 640, P / "04_traffic_segmentation_speed_anpr/data/traffic_overpass.mp4"),
    ("04 Plates (custom)", "plate_yolo26n.pt", 320, None),
    ("05 Workers (pose)", "yolo26n-pose.pt", 640, P / "05_robot_cell_safety/data/ur10_safety_zone.mp4"),
    ("05 Robots (YOLOE-26s)", "yoloe-26s-seg.pt", 640, P / "05_robot_cell_safety/data/ur10_safety_zone.mp4"),
]


def frame_from(video, size=(1280, 720)):
    if video is None or not Path(video).exists():
        return np.random.default_rng(0).integers(0, 255, (640, 640, 3), dtype=np.uint8)
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, 5000)
    ok, f = cap.read()
    return cv2.resize(f, size) if ok else frame_from(None)


def main(runs=40):
    rows = []
    cwd = os.getcwd()
    for name, w, sz, vid in CASES:
        path = MODELS / w
        if not path.exists():
            print("skip (missing)", w)
            continue
        if w.startswith("yoloe"):
            os.chdir(MODELS)
            m = YOLOE(str(path))
            m.set_classes(["robot arm", "industrial robot", "person"])
            os.chdir(cwd)
        else:
            m = YOLO(str(path))
        f = frame_from(vid)
        for _ in range(5):
            m(f, imgsz=sz, device=DEVICE, verbose=False)
        ts = []
        for _ in range(runs):
            s = time.perf_counter()
            m(f, imgsz=sz, device=DEVICE, verbose=False)
            ts.append(time.perf_counter() - s)
        ms = 1000 * float(np.median(ts))
        rows.append({"use_case": name, "model": w, "imgsz": sz, "latency_ms": round(ms, 1), "fps": round(1000 / ms, 1)})
        print(rows[-1])
    print("\n| Use case | Model | imgsz | CPU latency (ms) | Model FPS |\n|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['use_case']} | `{r['model']}` | {r['imgsz']} | {r['latency_ms']} | {r['fps']} |")
    (ROOT / "outputs").mkdir(exist_ok=True)
    (ROOT / "outputs" / "cpu_benchmark.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
