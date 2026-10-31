"""Incoming-goods / batch EL inspection: grade every EL image in a folder (or a single image / video).

Typical use: a pallet of modules arrives, the receiving team captures an EL image of each module with a
portable EL tester, and this script produces an acceptance report (CSV + JSON) with annotated evidence images.

Usage:
    python run.py --source /path/to/el_images/ --output outputs/delivery_2026-10-31
    python run.py --source data/el_cracked_module.mp4 --every 30 --conf 0.5      # sample frames from a video
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from config import DEVICE, MODELS  # noqa: E402
from draw import label  # noqa: E402

from inspect_line import CLASS_COLORS, grade  # noqa: E402
from ultralytics import YOLO  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_inputs(source, every):
    p = Path(source)
    if p.is_dir():
        for f in sorted(p.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                yield f.stem, cv2.imread(str(f))
    elif p.suffix.lower() in IMG_EXT:
        yield p.stem, cv2.imread(str(p))
    else:
        cap = cv2.VideoCapture(str(p))
        i = 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if i % every == 0:
                yield f"{p.stem}_f{i:05d}", f
            i += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--model", default=str(MODELS / "solar_defect_yolo26n.pt"))
    ap.add_argument("--rules", default="grading_rules.yaml")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--every", type=int, default=30, help="video: analyse every Nth frame")
    ap.add_argument("--output", default="outputs/batch_report")
    args = ap.parse_args()

    rules = yaml.safe_load(Path(args.rules).read_text())
    model = YOLO(args.model)
    out = Path(args.output)
    (out / "evidence").mkdir(parents=True, exist_ok=True)
    rows, grades = [], Counter()
    for name, img in iter_inputs(args.source, args.every):
        if img is None:
            continue
        gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)  # EL images are monochrome
        r = model(gray, imgsz=args.imgsz, conf=args.conf, device=DEVICE, verbose=False)[0]
        dets = [r.names[int(c)] for c in r.boxes.cls]
        g, reason = grade(dets, rules)
        grades[g] += 1
        vis = img.copy()
        for b, c, s in zip(r.boxes.xyxy.cpu().numpy().astype(int), r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
            col = CLASS_COLORS.get(r.names[c], (255, 255, 255))
            cv2.rectangle(vis, tuple(b[:2]), tuple(b[2:]), col, 2)
            label(vis, f"{r.names[c]} {s:.2f}", (b[0], b[1] - 2), color=col, scale=0.45, text_color=(20, 20, 20))
        label(vis, f"{name}: {g} ({reason})", (8, 30), color={"PASS": (80, 200, 60), "GRADE B": (0, 210, 255)}.get(g, (60, 60, 230)), scale=0.7)
        cv2.imwrite(str(out / "evidence" / f"{name}.jpg"), vis)
        rows.append({"id": name, "grade": g, "reason": reason, "defects": json.dumps(dict(Counter(dets)))})
        print(f"{name:<40} {g:<8} {reason}")

    with open(out / "report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "grade", "reason", "defects"])
        w.writeheader()
        w.writerows(rows)
    summary = {"inspected": len(rows), "grades": dict(grades),
               "accept_rate": round((grades["PASS"] + grades["GRADE B"]) / max(len(rows), 1), 3)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
