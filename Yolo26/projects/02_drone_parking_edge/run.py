"""Drone / overhead parking occupancy with YOLO26-OBB (rotated boxes for top-down vehicles).

Usage:
    python run.py --source data/parking_drone.mp4 --slots slot_rows.yaml
    python run.py --source rtsp://drone/stream --slots my_lot.json --model ../../models/yolo26n-obb_ncnn_model   # edge
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from config import DEVICE, model_path  # noqa: E402
from draw import BLUE, FONT, GREEN, GREY, RED, WHITE, YELLOW, bar, label, poly, watermark, with_sidebar  # noqa: E402
from registration import FrameRegistrar  # noqa: E402
from video_io import FPSMeter, VideoSink, VideoSource  # noqa: E402
from zones import poly_coverage  # noqa: E402

from slots import load_slots  # noqa: E402
from ultralytics import YOLO  # noqa: E402

VEHICLE_CLASSES = {"small vehicle", "large vehicle"}


class Slot:
    """Debounced occupancy. Becoming free needs a longer hold than becoming occupied, because a missed
    detection (shadow, dark car) is far more common than a phantom car."""

    def __init__(self, name, quad, hold_on, hold_off):
        self.name, self.ref_quad, self.quad = name, quad, quad
        self.hold = {True: hold_on, False: hold_off}
        self.occupied, self.pending, self.count, self.since = False, None, 0, 0.0

    def update(self, observed, t, immediate=False):
        if immediate:
            self.occupied, self.since = observed, t
            return None
        if observed == self.occupied:
            self.pending, self.count = None, 0
            return None
        if observed != self.pending:
            self.pending, self.count = observed, 0
        self.count += 1
        if self.count >= self.hold[observed]:
            self.occupied, self.pending, self.count, self.since = observed, None, 0, t
            return {"slot": self.name, "status": "occupied" if observed else "free", "t": round(t, 2)}
        return None


def sparkline(img, values, org, size, color=GREEN):
    x, y = org
    w, h = size
    cv2.rectangle(img, (x, y), (x + w, y + h), (45, 45, 45), -1)
    for g in (25, 50, 75):
        gy = int(y + h - g / 100 * h)
        cv2.line(img, (x, gy), (x + w, gy), (70, 70, 70), 1)
    if len(values) > 1:
        v = np.asarray(values, np.float32)
        pts = np.stack([x + np.linspace(0, w, len(v)), y + h - v / 100.0 * h], 1).astype(np.int32)
        cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)


def dashboard(canvas, x0, slots, moving, history, fps, inliers, t):
    occ = sum(s.occupied for s in slots)
    n = len(slots)
    y = 40
    cv2.putText(canvas, "SMART PARKING", (x0 + 20, y), FONT, 0.8, BLUE, 1, cv2.LINE_AA)
    cv2.putText(canvas, "drone view | live occupancy", (x0 + 20, y + 24), FONT, 0.45, GREY, 1, cv2.LINE_AA)
    y += 80
    cv2.putText(canvas, f"{n - occ}", (x0 + 20, y + 30), FONT, 1.8, GREEN, 2, cv2.LINE_AA)
    cv2.putText(canvas, "FREE", (x0 + 22, y + 56), FONT, 0.5, GREY, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{occ}", (x0 + 160, y + 30), FONT, 1.8, RED, 2, cv2.LINE_AA)
    cv2.putText(canvas, "OCCUPIED", (x0 + 162, y + 56), FONT, 0.5, GREY, 1, cv2.LINE_AA)
    y += 90
    rows = {}
    for s in slots:
        r = rows.setdefault(s.name[0], [0, 0])
        r[0] += s.occupied
        r[1] += 1
    for r, (o, tot) in rows.items():
        cv2.putText(canvas, f"Row {r}   {tot - o} free / {tot}", (x0 + 20, y), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
        bar(canvas, (x0 + 20, y + 8), (290, 10), o / tot, RED, bg=(60, 130, 60))
        y += 44
    y += 10
    cv2.putText(canvas, f"Occupancy {100 * occ / n:.0f}%  (trend)", (x0 + 20, y), FONT, 0.5, WHITE, 1, cv2.LINE_AA)
    sparkline(canvas, history, (x0 + 20, y + 10), (290, 70))
    y += 110
    cv2.putText(canvas, f"Vehicles in transit: {moving}", (x0 + 20, y), FONT, 0.5, YELLOW, 1, cv2.LINE_AA)
    y += 28
    recent = sorted((s for s in slots if s.since > 0), key=lambda s: -s.since)[:4]
    if recent:
        cv2.putText(canvas, "Recent changes", (x0 + 20, y), FONT, 0.5, GREY, 1, cv2.LINE_AA)
        y += 22
        for s in recent:
            col = RED if s.occupied else GREEN
            cv2.putText(canvas, f"{s.name}  {'occupied' if s.occupied else 'free':<9} {t - s.since:4.0f}s ago", (x0 + 20, y), FONT, 0.45, col, 1, cv2.LINE_AA)
            y += 20
    h = canvas.shape[0]
    cv2.putText(canvas, f"CPU {fps:4.1f} FPS | stab. inliers {inliers}", (x0 + 20, h - 20), FONT, 0.42, GREY, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--slots", default="slot_rows.yaml", help="slot_rows.yaml (parametric) or slots.json (annotated)")
    ap.add_argument("--model", default="yolo26n-obb.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--coverage", type=float, default=0.3, help="slot area fraction a vehicle must cover")
    ap.add_argument("--hold-on", type=int, default=10, help="frames before a slot is committed as occupied")
    ap.add_argument("--hold-off", type=int, default=45, help="frames before a slot is committed as free")
    ap.add_argument("--no-stabilize", action="store_true", help="disable drone drift compensation")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    model = YOLO(model_path(args.model), task="obb")
    src = VideoSource(args.source, max_seconds=args.max_seconds, width=args.width)
    out = Path(args.output or f"outputs/{Path(str(args.source)).stem}.mp4")
    sink = VideoSink(out, src.fps)
    slots, reg = None, None
    events, history = [], []
    meter = FPSMeter()
    t0 = time.time()

    for i, frame in enumerate(src):
        t = i / src.fps
        if slots is None:
            slots = [Slot(k, q, args.hold_on, args.hold_off) for k, q in load_slots(args.slots, frame.shape[1]).items()]
            reg = None if args.no_stabilize else FrameRegistrar(frame)
        if reg is not None:
            H = reg.update(frame)
            for s in slots:
                s.quad = FrameRegistrar.warp(s.ref_quad, H)

        res = model.track(frame, persist=True, imgsz=args.imgsz, conf=args.conf, device=DEVICE, verbose=False, tracker="bytetrack.yaml")[0]
        vis = frame.copy()

        quads, names, ids = [], [], []
        if res.obb is not None and len(res.obb):
            cls = res.obb.cls.int().cpu().tolist()
            pts = res.obb.xyxyxyxy.cpu().numpy()
            tid = res.obb.id.int().cpu().tolist() if res.obb.id is not None else [-1] * len(cls)
            for c, q, k in zip(cls, pts, tid):
                if res.names[c] in VEHICLE_CLASSES:
                    quads.append(q), names.append(res.names[c]), ids.append(k)

        in_slot = [False] * len(quads)
        for s in slots:
            covers = [poly_coverage(s.quad, q) for q in quads]
            best = int(np.argmax(covers)) if covers else -1
            observed = best >= 0 and covers[best] >= args.coverage
            if observed:
                in_slot[best] = True
            ev = s.update(observed, t, immediate=(i == 0))
            if ev:
                events.append(ev)
            poly(vis, s.quad, RED if s.occupied else GREEN, alpha=0.25 if s.occupied else 0.35, thickness=1)

        moving = 0
        for q, n, k, parked in zip(quads, names, ids, in_slot):
            cv2.polylines(vis, [q.astype(np.int32)], True, (230, 230, 230) if parked else YELLOW, 2, cv2.LINE_AA)
            if not parked:
                moving += 1
                label(vis, f"#{k} {'car' if n == 'small vehicle' else 'truck'}", q.min(0) - [0, 4], color=BLUE, scale=0.42)

        history.append(100.0 * sum(s.occupied for s in slots) / len(slots))
        history = history[-900:]
        fps = meter.tick()
        watermark(vis, "YOLO26-OBB | CPU | edge-ready: NCNN / IMX500 / TensorRT")
        canvas = with_sidebar(vis)
        dashboard(canvas, vis.shape[1], slots, moving, history, fps, reg.inliers if reg else 0, t)
        sink.write(canvas)
        if args.show:
            cv2.imshow("parking", canvas)
            if cv2.waitKey(1) == 27:
                break
    sink.close()

    summary = {
        "source": str(args.source), "slots": len(slots), "occupied_final": sum(s.occupied for s in slots),
        "avg_occupancy_pct": round(float(np.mean(history)), 1), "avg_cpu_fps": round(meter.mean, 1),
        "wall_time_s": round(time.time() - t0, 1),
        "slot_state": {s.name: "occupied" if s.occupied else "free" for s in slots},
        "events": events,
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    # The same payload is what an edge device would publish, e.g. MQTT topic parking/<lot_id>/status
    print(json.dumps({k: summary[k] for k in ("slots", "occupied_final", "avg_occupancy_pct", "avg_cpu_fps")}, indent=2))
    print(f"{len(events)} slot events -> {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
