"""End-of-line EL (electroluminescence) inspection station simulator for PV modules.

A real EL tester captures one high-resolution image per module while it stops on the conveyor.
This script reproduces that flow with HELD-OUT test images of PV-Multi-Defect (never seen in training):
rows x cols EL images = 1 module (default 1x1) -> tiled YOLO26 inference at native resolution ->
grading rules -> PASS / GRADE B / REJECT.
Because the test tiles have ground-truth labels, the grade predicted from the model is compared with the
grade derived from the annotations, giving a module-level verdict accuracy.

The same grading code is used for incoming-goods inspection (EL image of each delivered module).

Usage:
    python inspect_line.py --model ../../models/solar_defect_yolo26n.pt --modules 24
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from config import DEVICE, MODELS, ROOT  # noqa: E402
from draw import BLUE, FONT, GREEN, GREY, RED, WHITE, YELLOW, bar, label, watermark  # noqa: E402
from video_io import FPSMeter, VideoSink  # noqa: E402

from ultralytics import YOLO  # noqa: E402

CLASS_COLORS = {"black_border": (200, 120, 255), "broken": (60, 60, 240), "hot_spot": (0, 200, 255),
                "no_electricity": (255, 90, 90), "scratch": (80, 230, 120)}


def grade(defects, rules):
    """defects: list of class names. Returns (grade, reason)."""
    c = Counter(defects)
    crit = [k for k in rules["critical"] if c[k]]
    if crit:
        return "REJECT", f"critical: {', '.join(crit)}"
    minor = sum(c[k] for k in rules["minor"])
    if minor > rules["max_minor_for_b"]:
        return "REJECT", f"{minor} minor defects"
    if minor > 0:
        return "GRADE B", f"{minor} minor defect(s)"
    return "PASS", "no defects"


GRADE_COLOR = {"PASS": GREEN, "GRADE B": YELLOW, "REJECT": RED}


def load_gt(label_path, names):
    out = []
    if label_path.exists():
        for line in label_path.read_text().split("\n"):
            if line.strip():
                c, x, y, w, h = map(float, line.split())
                out.append((names[int(c)], x, y, w, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS / "solar_defect_yolo26n.pt"))
    ap.add_argument("--data", default=str(ROOT / "datasets" / "pv_defects.yaml"))
    ap.add_argument("--rules", default="grading_rules.yaml")
    ap.add_argument("--modules", type=int, default=20)
    ap.add_argument("--grid", default="1x1", help="EL images per module, rows x cols")
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--output", default="outputs/el_inspection_line.mp4")
    args = ap.parse_args()

    rules = yaml.safe_load(Path(args.rules).read_text())
    data = yaml.safe_load(Path(args.data).read_text())
    names = data["names"] if isinstance(data["names"], list) else [data["names"][i] for i in sorted(data["names"])]
    test_dir = Path(data["path"]) / data["test"]
    tiles = sorted(test_dir.glob("*.jpg"))
    random.Random(args.seed).shuffle(tiles)
    model = YOLO(args.model)

    W, H, SIDE = 1280, 720, 340
    rows, cols = map(int, args.grid.lower().split("x"))
    per_mod = rows * cols
    tile_px = min(440 // rows, 600 // cols)
    mod_w, mod_h = cols * tile_px, rows * tile_px
    fps = 30
    sink = VideoSink(args.output, fps)
    meter = FPSMeter()
    stats = Counter()
    defect_pareto = Counter()
    log, agree = [], 0
    station_x = (W - SIDE - mod_w) // 2
    y_mod = (H - mod_h) // 2 - 10

    def base_frame():
        f = np.full((H, W, 3), 30, np.uint8)
        # conveyor
        cv2.rectangle(f, (0, y_mod - 18), (W - SIDE, y_mod + mod_h + 18), (55, 55, 55), -1)
        for x in range(0, W - SIDE, 40):
            cv2.line(f, (x, y_mod + mod_h + 18), (x + 20, y_mod + mod_h + 4), (75, 75, 75), 2)
        # EL station hood
        cv2.rectangle(f, (station_x - 14, y_mod - 60), (station_x + mod_w + 14, y_mod - 22), (70, 70, 70), -1)
        cv2.putText(f, "EL STATION  |  YOLO26 defect inspection", (station_x, y_mod - 34), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
        return f

    def sidebar(f, current=None):
        x0 = W - SIDE
        cv2.rectangle(f, (x0, 0), (W, H), (24, 24, 24), -1)
        cv2.line(f, (x0, 0), (x0, H), (70, 70, 70), 2)
        x = x0 + 20
        y = 40
        cv2.putText(f, "PV MODULE QC", (x, y), FONT, 0.8, BLUE, 1, cv2.LINE_AA)
        cv2.putText(f, "end-of-line EL inspection", (x, y + 24), FONT, 0.45, GREY, 1, cv2.LINE_AA)
        y += 66
        n = sum(stats.values())
        cv2.putText(f, f"Modules inspected: {n}", (x, y), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
        y += 30
        for g in ("PASS", "GRADE B", "REJECT"):
            frac = stats[g] / n if n else 0
            cv2.putText(f, f"{g:<8} {stats[g]:>3}  ({100 * frac:.0f}%)", (x, y), FONT, 0.5, GRADE_COLOR[g], 1, cv2.LINE_AA)
            bar(f, (x, y + 7), (290, 8), frac, GRADE_COLOR[g])
            y += 36
        y += 6
        cv2.putText(f, "Defect Pareto", (x, y), FONT, 0.5, GREY, 1, cv2.LINE_AA)
        y += 10
        top = defect_pareto.most_common()
        mx = top[0][1] if top else 1
        for k, v in top[:5]:
            y += 24
            cv2.putText(f, f"{k:<15}{v:>4}", (x, y), FONT, 0.45, CLASS_COLORS.get(k, WHITE), 1, cv2.LINE_AA)
            bar(f, (x + 190, y - 10), (100, 10), v / mx, CLASS_COLORS.get(k, WHITE))
        y += 36
        acc = 100 * agree / n if n else 0
        cv2.putText(f, f"Verdict vs ground truth: {acc:.0f}%", (x, y), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        cv2.putText(f, "(held-out test modules)", (x, y + 20), FONT, 0.4, GREY, 1, cv2.LINE_AA)
        y += 50
        cv2.putText(f, "Recent", (x, y), FONT, 0.5, GREY, 1, cv2.LINE_AA)
        for e in log[-5:][::-1]:
            y += 22
            cv2.putText(f, f"{e['module_id']}  {e['grade']}", (x, y), FONT, 0.45, GRADE_COLOR[e["grade"]], 1, cv2.LINE_AA)
        cv2.putText(f, f"CPU inference {meter.mean:4.1f} tiles/s", (x, H - 18), FONT, 0.42, GREY, 1, cv2.LINE_AA)

    n_mod = min(args.modules, len(tiles) // per_mod)
    for m in range(n_mod):
        group = tiles[m * per_mod:(m + 1) * per_mod]
        module = np.zeros((rows * 600, cols * 600, 3), np.uint8)
        dets, gt_defects = [], []
        for j, tp in enumerate(group):
            r_, c_ = divmod(j, cols)
            img = cv2.imread(str(tp))
            img = cv2.resize(img, (600, 600))
            module[r_ * 600:(r_ + 1) * 600, c_ * 600:(c_ + 1) * 600] = img
            meter.tick()
            res = model(img, imgsz=640, conf=args.conf, device=DEVICE, verbose=False)[0]
            for b, c, s in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.int().cpu().tolist(), res.boxes.conf.cpu().tolist()):
                dets.append((names[c], s, b + [c_ * 600, r_ * 600, c_ * 600, r_ * 600]))
            gt_defects += [g[0] for g in load_gt(tp.parent.parent.parent / "labels" / tp.parent.name / (tp.stem + ".txt"), names)]
        g_pred, reason = grade([d[0] for d in dets], rules)
        g_gt, _ = grade(gt_defects, rules)
        mod_small = cv2.resize(module, (mod_w, mod_h), interpolation=cv2.INTER_AREA)
        # cell grid lines to read like a module
        for k in range(1, cols):
            cv2.line(mod_small, (k * tile_px, 0), (k * tile_px, mod_h), (20, 20, 20), 2)
        for k in range(1, rows):
            cv2.line(mod_small, (0, k * tile_px), (mod_w, k * tile_px), (20, 20, 20), 2)
        annotated = mod_small.copy()
        sc = tile_px / 600
        for name, s, b in dets:
            p1, p2 = (b[:2] * sc).astype(int), (b[2:] * sc).astype(int)
            cv2.rectangle(annotated, tuple(p1), tuple(p2), CLASS_COLORS[name], 2)
        mid = f"PV-{m + 1:04d}"

        # timeline: slide in -> capture flash -> reveal detections -> verdict hold -> slide out (~2.2 s per module)
        start_x, stop_x, end_x = -mod_w, station_x, W - SIDE + 10
        for k in range(12):
            f = base_frame()
            x = int(start_x + (stop_x - start_x) * (k + 1) / 12)
            _blit(f, mod_small, x, y_mod, W - SIDE)
            sidebar(f)
            sink.write(f)
        for k in range(4):
            f = base_frame()
            _blit(f, mod_small, stop_x, y_mod, W - SIDE)
            cv2.rectangle(f, (stop_x - 6, y_mod - 6), (stop_x + mod_w + 6, y_mod + mod_h + 6), (255, 255, 255), 3)
            label(f, "CAPTURING EL IMAGE...", (stop_x, y_mod + mod_h + 50), color=BLUE, scale=0.6)
            sidebar(f)
            sink.write(f)
        for k in range(14):
            f = base_frame()
            reveal = int(mod_w * (k + 1) / 14)
            comp = mod_small.copy()
            comp[:, :reveal] = annotated[:, :reveal]
            _blit(f, comp, stop_x, y_mod, W - SIDE)
            cv2.line(f, (stop_x + reveal, y_mod - 4), (stop_x + reveal, y_mod + mod_h + 4), (255, 220, 0), 2)
            label(f, f"{mid}  analysing...", (stop_x, y_mod + mod_h + 50), color=BLUE, scale=0.6)
            sidebar(f)
            sink.write(f)
        stats[g_pred] += 1
        agree += int(g_pred == g_gt)
        for d in dets:
            defect_pareto[d[0]] += 1
        log.append({"module_id": mid, "grade": g_pred, "reason": reason, "gt_grade": g_gt,
                    "defects": dict(Counter(d[0] for d in dets)), "tiles": [t.name for t in group]})
        for k in range(26):
            f = base_frame()
            _blit(f, annotated, stop_x, y_mod, W - SIDE)
            col = GRADE_COLOR[g_pred]
            cv2.rectangle(f, (stop_x - 6, y_mod - 6), (stop_x + mod_w + 6, y_mod + mod_h + 6), col, 4)
            label(f, f"{mid}: {g_pred}  -  {reason}", (stop_x, y_mod + mod_h + 50), color=col, scale=0.7, thickness=2)
            cnt = Counter(d[0] for d in dets)
            lx = stop_x
            for name, v in cnt.items():
                label(f, f"{name} x{v}", (lx, y_mod + mod_h + 88), color=CLASS_COLORS[name], scale=0.5, text_color=(20, 20, 20))
                lx += 18 + cv2.getTextSize(f"{name} x{v}", FONT, 0.5, 1)[0][0]
            sidebar(f)
            sink.write(f)
        for k in range(12):
            f = base_frame()
            x = int(stop_x + (end_x - stop_x) * (k + 1) / 12)
            _blit(f, annotated, x, y_mod, W - SIDE)
            sidebar(f)
            sink.write(f)
        print(f"{mid}: pred={g_pred:<8} gt={g_gt:<8} {reason}")

    f = base_frame()
    sidebar(f)
    watermark(f, "YOLO26n fine-tuned on PV-Multi-Defect | CPU")
    for _ in range(45):
        sink.write(f)
    sink.close()

    out = Path(args.output)
    with open(out.with_suffix(".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["module_id", "grade", "reason", "gt_grade", "defects"])
        w.writeheader()
        for e in log:
            w.writerow({k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in e.items() if k != "tiles"})
    summary = {"modules": n_mod, "grades": dict(stats), "verdict_accuracy_vs_gt": round(agree / max(n_mod, 1), 3),
               "defect_pareto": dict(defect_pareto), "cpu_tiles_per_s": round(meter.mean, 1)}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _blit(dst, src, x, y, max_x):
    """Paste src at (x, y), clipped to [0, max_x)."""
    h, w = src.shape[:2]
    x0, x1 = max(0, x), min(max_x, x + w)
    if x1 <= x0:
        return
    dst[y:y + h, x0:x1] = src[:, x0 - x:x1 - x]


if __name__ == "__main__":
    main()
