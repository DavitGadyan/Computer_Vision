"""Traffic analytics: YOLO26 instance segmentation + tracking + lane occupancy + speed + ANPR.

Pipeline per frame
  1. yolo26n-seg (+ByteTrack) -> vehicle masks with persistent IDs
  2. lane assignment from the ground-contact point, line-crossing counts per lane/class
  3. mask pixels inside each lane -> road-space occupancy % -> congestion level
  4. homography image->road plane (metres) -> per-track speed (km/h), speeding flag
  5. close vehicles: fine-tuned yolo26n plate detector on the vehicle crop -> OCR -> per-track vote

Usage:
    python run.py --source data/traffic_overpass.mp4 --scene traffic_overpass.yaml
    python run.py --source data/traffic_overpass.mp4 --scene traffic_overpass.yaml --blur-plates   # GDPR-friendly output
"""
import argparse
import csv
import json
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from config import DEVICE, MODELS, model_path  # noqa: E402
from draw import BLUE, FONT, GREEN, GREY, RED, WHITE, YELLOW, bar, label, watermark, with_sidebar  # noqa: E402
from video_io import FPSMeter, VideoSink, VideoSource  # noqa: E402
from zones import point_in_poly  # noqa: E402

from ultralytics import YOLO  # noqa: E402

VEHICLES = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
LANE_COLORS = [(235, 150, 40), (40, 160, 245), (120, 220, 120), (200, 120, 230)]


@dataclass
class Track:
    tid: int
    cls: str
    lane: str | None = None
    pts: deque = field(default_factory=lambda: deque(maxlen=20))  # (t, x_m, y_m)
    speed: float | None = None
    max_speed: float = 0.0
    speeds: list = field(default_factory=list)
    votes: dict = field(default_factory=lambda: defaultdict(float))
    plate_crop: np.ndarray | None = None
    plate_conf: float = 0.0
    counted: bool = False
    last_y: float | None = None
    first_t: float = 0.0
    last_t: float = 0.0
    ocr_tries: int = 0

    @property
    def plate(self):
        if not self.votes:
            return None
        text, score = max(self.votes.items(), key=lambda kv: kv[1])
        return text if score >= 1.6 else None  # needs ~2 confident reads


class SpeedEstimator:
    def __init__(self, cfg):
        src = np.float32(cfg["source"])
        w, length = cfg["width_m"], cfg["length_m"]
        dst = np.float32([[0, 0], [w, 0], [w, length], [0, length]])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.region = src

    def to_road(self, pt):
        p = cv2.perspectiveTransform(np.float32([[pt]]), self.M)[0, 0]
        return float(p[0]), float(p[1])

    @staticmethod
    def update(track: Track, min_span=0.4):
        """Least-squares slope of road position over the last ~0.7 s -> km/h."""
        if len(track.pts) < 6:
            return None
        t = np.array([p[0] for p in track.pts])
        if t[-1] - t[0] < min_span:
            return None
        y = np.array([p[2] for p in track.pts])
        x = np.array([p[1] for p in track.pts])
        vy, vx = np.polyfit(t, y, 1)[0], np.polyfit(t, x, 1)[0]
        kmh = 3.6 * float(np.hypot(vx, vy))
        track.speed = kmh if track.speed is None else 0.7 * track.speed + 0.3 * kmh
        return track.speed


class PlateReader:
    def __init__(self, det_weights, ocr_model="global-plates-mobile-vit-v2-model"):
        from fast_plate_ocr import LicensePlateRecognizer

        self.det = YOLO(det_weights)
        self.ocr = LicensePlateRecognizer(ocr_model, device="cpu")
        self.gray = self.ocr.config.image_color_mode == "grayscale"

    def read(self, vehicle_crop):
        r = self.det(vehicle_crop, imgsz=320, conf=0.35, device=DEVICE, verbose=False)[0]
        if not len(r.boxes):
            return None
        i = int(r.boxes.conf.argmax())
        x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy().astype(int)
        pad_x, pad_y = int(0.06 * (x2 - x1)), int(0.12 * (y2 - y1))
        h, w = vehicle_crop.shape[:2]
        plate = vehicle_crop[max(0, y1 - pad_y):min(h, y2 + pad_y), max(0, x1 - pad_x):min(w, x2 + pad_x)]
        if plate.shape[0] < 8 or plate.shape[1] < 25:
            return None
        x = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY if self.gray else cv2.COLOR_BGR2RGB)
        pred = self.ocr.run(x, return_confidence=True)[0]
        conf = float(np.mean(pred.char_probs)) if len(pred.char_probs) else 0.0
        return pred.plate, conf, plate, (x1, y1, x2, y2), float(r.boxes.conf[i])


def congestion(occ_pct, mean_speed):
    if occ_pct > 35 or (mean_speed is not None and mean_speed < 15):
        return "HEAVY", RED
    if occ_pct > 18 or (mean_speed is not None and mean_speed < 35):
        return "MODERATE", YELLOW
    return "FREE FLOW", GREEN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--scene", required=True, help="scene calibration yaml (lanes, speed homography, count line)")
    ap.add_argument("--model", default="yolo26n-seg.pt")
    ap.add_argument("--plate-model", default=str(MODELS / "plate_yolo26n.pt"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--blur-plates", action="store_true", help="anonymise plates in the output video")
    ap.add_argument("--out-width", type=int, default=1280)
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    scene = yaml.safe_load(Path(args.scene).read_text())
    lanes = {k: np.float32(v) for k, v in scene["lanes"].items()}
    lane_color = {k: LANE_COLORS[i % len(LANE_COLORS)] for i, k in enumerate(lanes)}
    speed_est = SpeedEstimator(scene["speed"])
    limit = scene["speed"].get("limit_kmh", 50)
    count_y = scene["count_line_y"]
    anpr_cfg = scene.get("anpr", {})
    plate_re = re.compile(anpr_cfg["plate_regex"]) if anpr_cfg.get("plate_regex") else None

    seg = YOLO(model_path(args.model))
    plates = PlateReader(args.plate_model) if Path(args.plate_model).exists() else None
    if plates is None:
        print(f"[warn] plate model {args.plate_model} not found - ANPR disabled (train it with common/train.py)")

    src = VideoSource(args.source, max_seconds=args.max_seconds)
    out = Path(args.output or f"outputs/{Path(str(args.source)).stem}.mp4")
    sink = VideoSink(out, src.fps)
    tracks: dict[int, Track] = {}
    counts = {k: defaultdict(int) for k in lanes}
    anpr_log = []
    meter = FPSMeter()
    lane_masks = None

    for i, frame in enumerate(src):
        t = i / src.fps
        H, W = frame.shape[:2]
        if lane_masks is None:  # quarter-resolution lane masks for occupancy
            lane_masks = {}
            for k, p in lanes.items():
                m = np.zeros((H // 4, W // 4), np.uint8)
                cv2.fillPoly(m, [(p / 4).astype(np.int32)], 1)
                lane_masks[k] = m.astype(bool)
        res = seg.track(frame, persist=True, imgsz=args.imgsz, conf=0.3, classes=list(VEHICLES), device=DEVICE,
                        verbose=False, tracker="bytetrack.yaml", retina_masks=False)[0]
        vis = frame.copy()
        overlay = vis.copy()
        veh_mask = np.zeros((H // 4, W // 4), np.uint8)

        if res.boxes.id is not None and res.masks is not None:
            boxes = res.boxes.xyxy.cpu().numpy()
            ids = res.boxes.id.int().cpu().tolist()
            cls = res.boxes.cls.int().cpu().tolist()
            polys = res.masks.xy
            for box, tid, c, pg in zip(boxes, ids, cls, polys):
                x1, y1, x2, y2 = box
                ground = ((x1 + x2) / 2, y2)
                tr = tracks.setdefault(tid, Track(tid, VEHICLES[c], first_t=t))
                tr.last_t = t
                tr.lane = next((k for k, p in lanes.items() if point_in_poly(ground, p)), tr.lane)
                col = lane_color.get(tr.lane, (180, 180, 180))
                if len(pg) >= 3:
                    cv2.fillPoly(overlay, [pg.astype(np.int32)], col)
                    cv2.polylines(vis, [pg.astype(np.int32)], True, col, 2, cv2.LINE_AA)
                    cv2.fillPoly(veh_mask, [(pg / 4).astype(np.int32)], 1)

                # counting on line crossing (top -> bottom = towards camera)
                if tr.last_y is not None and tr.last_y < count_y <= y2 and not tr.counted and tr.lane:
                    counts[tr.lane][tr.cls] += 1
                    tr.counted = True
                tr.last_y = y2

                # speed inside the calibrated region only; skip boxes clipped by the frame edge (ground point unreliable)
                if cv2.pointPolygonTest(speed_est.region, ground, False) >= 0 and y2 < H - 6:
                    xm, ym = speed_est.to_road(ground)
                    tr.pts.append((t, xm, ym))
                    if SpeedEstimator.update(tr) is not None:
                        tr.speeds.append(tr.speed)
                        if len(tr.speeds) >= 10:  # ignore the warm-up of a new track
                            tr.max_speed = max(tr.max_speed, float(np.median(tr.speeds[-10:])))

                # ANPR on close, wide vehicles; stop once the plate is confirmed
                if (plates and y2 > anpr_cfg.get("min_y2", 0) and (x2 - x1) > anpr_cfg.get("min_width", 0)
                        and (tr.plate is None or tr.ocr_tries < 8) and i % 2 == 0):
                    vx1, vy1, vx2, vy2 = map(int, (max(0, x1), max(0, y1), min(W, x2), min(H, y2)))
                    crop = frame[vy1:vy2, vx1:vx2]
                    r = plates.read(crop)
                    tr.ocr_tries += 1
                    if r:
                        text, conf, pcrop, (px1, py1, px2, py2), _ = r
                        if 5 <= len(text) <= 9 and conf > 0.6:
                            # reads that don't match the regional format still count, but much less
                            tr.votes[text] += conf if plate_re is None or plate_re.fullmatch(text) else 0.3 * conf
                            if conf > tr.plate_conf:
                                tr.plate_conf, tr.plate_crop = conf, pcrop.copy()
                        if args.blur_plates:
                            roi = vis[vy1 + py1:vy1 + py2, vx1 + px1:vx1 + px2]
                            if roi.size:
                                vis[vy1 + py1:vy1 + py2, vx1 + px1:vx1 + px2] = cv2.GaussianBlur(roi, (31, 31), 0)
                        else:
                            cv2.rectangle(vis, (vx1 + px1, vy1 + py1), (vx1 + px2, vy1 + py2), YELLOW, 2)
                        if tr.plate and not any(e["track_id"] == tid for e in anpr_log):
                            anpr_log.append({"track_id": tid, "plate": tr.plate, "class": tr.cls, "lane": tr.lane, "t": round(t, 2)})

                # label
                txt = f"#{tid} {tr.cls}"
                if tr.speed is not None:
                    txt += f" {tr.speed:.0f} km/h"
                if tr.plate and not args.blur_plates:
                    txt += f" | {tr.plate}"
                speeding = len(tr.speeds) >= 10 and tr.speed > limit
                label(vis, txt, (x1, y1 - 4), color=RED if speeding else col, scale=0.85, thickness=2, pad=6)

        cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)
        for k, p in lanes.items():
            cv2.polylines(vis, [p.astype(np.int32)], True, lane_color[k], 1, cv2.LINE_AA)
        cv2.line(vis, (int(lanes[next(iter(lanes))][:, 0].min()), count_y), (int(max(p[:, 0].max() for p in lanes.values())), count_y), WHITE, 2)

        # lane occupancy from segmentation masks
        occ = {k: 100.0 * (veh_mask.astype(bool) & m).sum() / m.sum() for k, m in lane_masks.items()}
        live = [tr for tr in tracks.values() if tr.last_t == t and tr.speed is not None]
        lane_speed = {k: (np.mean([tr.speed for tr in live if tr.lane == k]) if any(tr.lane == k for tr in live) else None) for k in lanes}
        all_speed = np.mean([tr.speed for tr in live]) if live else None
        level, lcol = congestion(np.mean(list(occ.values())), all_speed)

        fps = meter.tick()
        vis = cv2.resize(vis, (args.out_width, int(H * args.out_width / W)), interpolation=cv2.INTER_AREA)
        watermark(vis, "YOLO26-seg + custom YOLO26 plate model | CPU")
        canvas = with_sidebar(vis, width=360)
        x0, y = vis.shape[1] + 20, 38
        cv2.putText(canvas, "TRAFFIC ANALYTICS", (x0, y), FONT, 0.8, BLUE, 1, cv2.LINE_AA)
        y += 36
        cv2.putText(canvas, f"Congestion: {level}", (x0, y), FONT, 0.6, lcol, 1, cv2.LINE_AA)
        y += 34
        for k in lanes:
            total = sum(counts[k].values())
            sp = f"{lane_speed[k]:.0f} km/h" if lane_speed[k] is not None else "--"
            cv2.putText(canvas, f"{k}: {total} veh | {sp}", (x0, y), FONT, 0.52, lane_color[k], 1, cv2.LINE_AA)
            y += 16
            bar(canvas, (x0, y), (310, 9), occ[k] / 60.0, lane_color[k])
            cv2.putText(canvas, f"occupancy {occ[k]:.0f}%", (x0 + 200, y - 4), FONT, 0.38, GREY, 1, cv2.LINE_AA)
            y += 30
        by_cls = defaultdict(int)
        for k in lanes:
            for c, n in counts[k].items():
                by_cls[c] += n
        cv2.putText(canvas, "  ".join(f"{c}s {n}" for c, n in sorted(by_cls.items())) or "counting...", (x0, y), FONT, 0.48, WHITE, 1, cv2.LINE_AA)
        y += 24
        n_fast = sum(1 for tr in tracks.values() if tr.max_speed > limit)
        cv2.putText(canvas, f"Over {limit} km/h: {n_fast}", (x0, y), FONT, 0.48, RED if n_fast else GREY, 1, cv2.LINE_AA)
        y += 34
        cv2.putText(canvas, "ANPR - plates read", (x0, y), FONT, 0.55, YELLOW, 1, cv2.LINE_AA)
        y += 10
        for e in anpr_log[-7:][::-1]:
            tr = tracks[e["track_id"]]
            if tr.plate_crop is not None and not args.blur_plates:
                pc = cv2.resize(tr.plate_crop, (110, max(16, int(110 * tr.plate_crop.shape[0] / tr.plate_crop.shape[1]))))[:34]
                canvas[y + 6:y + 6 + pc.shape[0], x0:x0 + 110] = pc
            txt = "*** blurred ***" if args.blur_plates else (tr.plate or e["plate"])
            cv2.putText(canvas, f"{txt}", (x0 + 122, y + 24), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"#{e['track_id']} {e['class']}", (x0 + 250, y + 24), FONT, 0.4, GREY, 1, cv2.LINE_AA)
            y += 42
        cv2.putText(canvas, f"CPU {fps:4.1f} FPS (1080p input)", (x0, canvas.shape[0] - 18), FONT, 0.42, GREY, 1, cv2.LINE_AA)
        sink.write(canvas)
        if args.show:
            cv2.imshow("traffic", canvas)
            if cv2.waitKey(1) == 27:
                break
    sink.close()

    rows = []
    for tr in tracks.values():
        if tr.last_t - tr.first_t < 0.5:
            continue
        rows.append({"track_id": tr.tid, "class": tr.cls, "lane": tr.lane,
                     "avg_speed_kmh": round(float(np.median(tr.speeds)), 1) if tr.speeds else "",
                     "max_speed_kmh": round(tr.max_speed, 1) if tr.speeds else "",
                     "plate": tr.plate or "", "first_seen_s": round(tr.first_t, 2), "last_seen_s": round(tr.last_t, 2)})
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["track_id"])
        w.writeheader()
        w.writerows(rows)
    for e in anpr_log:  # final consolidated vote
        e["plate"] = tracks[e["track_id"]].plate or e["plate"]
    summary = {"source": str(args.source), "avg_cpu_fps": round(meter.mean, 1),
               "counts": {k: dict(v) for k, v in counts.items()}, "plates_read": len(anpr_log),
               "vehicles_tracked": len(rows), "anpr": anpr_log}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "anpr"}, indent=2))


if __name__ == "__main__":
    main()
