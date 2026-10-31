"""AI workout monitor: multi-person rep counting + form feedback with YOLO26-pose.

Usage:
    python run.py --source data/pushups.mp4 --output outputs/pushups.mp4
    python run.py --source 0 --show                      # webcam
    python run.py --source data/squats_gym.mp4 --exercise squat
"""
import argparse
import json
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from config import DEVICE, model_path  # noqa: E402
from draw import BLUE, GREEN, RED, WHITE, YELLOW, color_for_id, label, panel, watermark  # noqa: E402
from video_io import FPSMeter, VideoSink, VideoSource  # noqa: E402

from ultralytics import YOLO  # noqa: E402

# COCO-17 keypoint indices
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP, L_KN, R_KN, L_AN, R_AN = 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
SIDES = {"left": (L_SH, L_EL, L_WR, L_HIP, L_KN, L_AN), "right": (R_SH, R_EL, R_WR, R_HIP, R_KN, R_AN)}

# Rep thresholds (degrees). "down" must be reached for a full rep, "up" closes the rep.
RULES = {
    "squat": dict(joint="knee", up=160, down=100, partial=125),
    "pushup": dict(joint="elbow", up=150, down=95, partial=120),
    "curl": dict(joint="elbow_curl", up=150, down=55, partial=80),
}


def angle(a, b, c):
    """Angle ABC in degrees."""
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


@dataclass
class Athlete:
    tid: int
    exercise: str | None = None
    stage: str = "up"
    reps: int = 0
    good_reps: int = 0
    min_angle: float = 180.0
    smooth: dict = field(default_factory=dict)
    history: deque = field(default_factory=lambda: deque(maxlen=60))
    torso_hist: deque = field(default_factory=lambda: deque(maxlen=30))
    feedback: str = ""
    feedback_ttl: int = 0
    issues: dict = field(default_factory=dict)
    frames: int = 0

    def ema(self, key, value, alpha=0.35):
        self.smooth[key] = value if key not in self.smooth else (1 - alpha) * self.smooth[key] + alpha * value
        return self.smooth[key]

    def flag(self, msg, ttl=30):
        if msg != self.feedback or self.feedback_ttl <= 0:  # count each occurrence once, not every frame
            self.issues[msg] = self.issues.get(msg, 0) + 1
        self.feedback, self.feedback_ttl = msg, ttl


def joint_angles(kpts, conf, min_conf=0.35):
    """Compute angles from the more visible body side. Returns (angles dict, side points) or None."""
    best = max(SIDES, key=lambda s: conf[list(SIDES[s])].mean())
    sh, el, wr, hip, kn, an = SIDES[best]
    if conf[[sh, hip]].min() < min_conf:
        return None
    p = kpts
    ang = {}
    if conf[[sh, el, wr]].min() >= min_conf:
        ang["elbow"] = ang["elbow_curl"] = angle(p[sh], p[el], p[wr])
    if conf[[hip, kn, an]].min() >= min_conf:
        ang["knee"] = angle(p[hip], p[kn], p[an])
        ang["body_line"] = angle(p[sh], p[hip], p[an])
    if conf[[sh, hip, kn]].min() >= min_conf:
        ang["hip"] = angle(p[sh], p[hip], p[kn])
    torso = p[sh] - p[hip]
    ang["torso_tilt"] = float(np.degrees(np.arctan2(abs(torso[0]), abs(torso[1]) + 1e-6)))  # 0 = upright, 90 = horizontal
    if conf[[sh, el]].min() >= min_conf:
        upper = p[el] - p[sh]
        ang["upper_arm_swing"] = float(np.degrees(np.arctan2(abs(upper[0]), abs(upper[1]) + 1e-6)))
    pts = {"shoulder": p[sh], "elbow": p[el], "wrist": p[wr], "hip": p[hip], "knee": p[kn], "ankle": p[an]}
    return ang, pts


def classify(ath: Athlete, ang):
    """Auto-detect exercise from body orientation and which joint is moving."""
    ath.torso_hist.append(ang["torso_tilt"])
    ath.history.append((ang.get("knee"), ang.get("elbow")))
    if len(ath.history) < 20:
        return None
    if np.median(ath.torso_hist) > 55:
        return "pushup"
    knees = [k for k, _ in ath.history if k is not None]
    elbows = [e for _, e in ath.history if e is not None]
    knee_rng = np.ptp(knees) if len(knees) > 10 else 0
    elbow_rng = np.ptp(elbows) if len(elbows) > 10 else 0
    if knee_rng > 35:
        return "squat"
    if elbow_rng > 60 and knee_rng < 20:
        return "curl"
    return None


def update_reps(ath: Athlete, ang):
    rule = RULES[ath.exercise]
    raw = ang.get(rule["joint"])
    if raw is None:
        return None
    a = ath.ema("main", raw)
    ath.min_angle = min(ath.min_angle, a)

    # Continuous form checks (only while moving down/at the bottom)
    if ath.exercise == "pushup" and "body_line" in ang and ath.stage == "down":
        if ang["body_line"] < 150:
            ath.flag("Keep body straight (hips)")
    if ath.exercise == "squat" and "hip" in ang and ath.stage == "down":
        if ang["torso_tilt"] > 50:
            ath.flag("Chest up - torso leaning")
    if ath.exercise == "curl" and ang.get("upper_arm_swing", 0) > 35:
        ath.flag("Pin elbows - no swinging")

    if ath.stage == "up" and a < rule["partial"]:
        ath.stage = "down"
        ath.min_angle = a
    elif ath.stage == "down" and a > rule["up"]:
        ath.stage = "up"
        ath.reps += 1
        if ath.min_angle <= rule["down"]:
            ath.good_reps += 1
            ath.flag("Good rep!", ttl=20)
        else:
            ath.flag({"squat": "Go deeper", "pushup": "Lower your chest", "curl": "Full range of motion"}[ath.exercise])
        ath.min_angle = 180.0
    return a


def draw_arc(img, center, a_pt, c_pt, ang, color):
    """Draw the measured joint angle as an arc + value."""
    c = tuple(int(v) for v in center)
    v1, v2 = a_pt - center, c_pt - center
    t1, t2 = np.degrees(np.arctan2(v1[1], v1[0])), np.degrees(np.arctan2(v2[1], v2[0]))
    if (t2 - t1) % 360 > 180:
        t1, t2 = t2, t1
    cv2.ellipse(img, c, (28, 28), 0, t1, t1 + (t2 - t1) % 360, color, 3, cv2.LINE_AA)
    label(img, f"{int(ang)} deg", (c[0] + 30, c[1] - 8), color=color, scale=0.45)


SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def draw_skeleton(img, kpts, conf, color, min_conf=0.35):
    for i, j in SKELETON:
        if conf[i] > min_conf and conf[j] > min_conf:
            cv2.line(img, tuple(kpts[i].astype(int)), tuple(kpts[j].astype(int)), color, 3, cv2.LINE_AA)
    for k in range(5, 17):
        if conf[k] > min_conf:
            cv2.circle(img, tuple(kpts[k].astype(int)), 5, WHITE, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(kpts[k].astype(int)), 5, color, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--model", default="yolo26n-pose.pt")
    ap.add_argument("--exercise", choices=list(RULES), default=None, help="force exercise (default: auto-detect)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--min-height", type=float, default=0.3, help="ignore people smaller than this fraction of frame height (background)")
    ap.add_argument("--min-rel-height", type=float, default=0.6, help="ignore people smaller than this fraction of the tallest person (mirror reflections)")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    model = YOLO(model_path(args.model))
    src = VideoSource(args.source, max_seconds=args.max_seconds, width=args.width)
    out = Path(args.output or f"outputs/{Path(str(args.source)).stem}.mp4")
    sink = VideoSink(out, src.fps)
    athletes: dict[int, Athlete] = {}
    meter = FPSMeter()

    for frame in src:
        res = model.track(frame, persist=True, device=DEVICE, imgsz=args.imgsz, conf=0.4, verbose=False, tracker="bytetrack.yaml")[0]
        vis = frame.copy()
        if res.keypoints is not None and res.boxes.id is not None:
            kp = res.keypoints.xy.cpu().numpy()
            kc = res.keypoints.conf.cpu().numpy()
            ids = res.boxes.id.int().cpu().tolist()
            boxes = res.boxes.xyxy.cpu().numpy()
            tallest = (boxes[:, 3] - boxes[:, 1]).max()
            for k, c, tid, box in zip(kp, kc, ids, boxes):
                h = box[3] - box[1]
                if h < args.min_height * frame.shape[0] or h < args.min_rel_height * tallest:
                    continue
                ath = athletes.setdefault(tid, Athlete(tid, exercise=args.exercise))
                ath.frames += 1
                color = color_for_id(tid)
                draw_skeleton(vis, k, c, color)
                res_ang = joint_angles(k, c)
                if res_ang is None:
                    continue
                ang, pts = res_ang
                if ath.exercise is None:
                    ath.exercise = classify(ath, ang)
                if ath.exercise:
                    a = update_reps(ath, ang)
                    rule = RULES[ath.exercise]
                    if a is not None:
                        jc = GREEN if a <= rule["down"] else (YELLOW if a <= rule["partial"] else BLUE)
                        if rule["joint"] == "knee":
                            draw_arc(vis, pts["knee"], pts["hip"], pts["ankle"], a, jc)
                        else:
                            draw_arc(vis, pts["elbow"], pts["shoulder"], pts["wrist"], a, jc)
                x1, y1 = int(box[0]), int(box[1])
                if y1 < 150 and x1 < 420:  # keep labels clear of the HUD panel / frame edge
                    y1 = int(box[3]) - 40
                title = f"#{tid} {ath.exercise.upper() if ath.exercise else 'detecting...'}"
                label(vis, f"{title}  reps {ath.reps}  ({ath.stage})", (x1, y1 - 4), color=color, scale=0.55)
                if ath.feedback_ttl > 0:
                    fc = GREEN if ath.feedback.startswith("Good") else RED
                    label(vis, ath.feedback, (x1, y1 + 26), color=fc, scale=0.6)
                    ath.feedback_ttl -= 1

        fps = meter.tick()
        active = [a for a in athletes.values() if a.exercise and a.frames > 15]
        lines = [f"{'#' + str(a.tid):>4} {a.exercise:<7} reps {a.reps:>2}  good {a.good_reps:>2}" for a in active] or ["waiting for athlete..."]
        lines.append((f"CPU {fps:4.1f} FPS", (180, 180, 180)))
        panel(vis, lines, title="AI WORKOUT MONITOR", scale=0.6)
        watermark(vis, "YOLO26-pose | CPU | form & rep analytics")
        sink.write(vis)
        if args.show:
            cv2.imshow("workout", vis)
            if cv2.waitKey(1) == 27:
                break
    sink.close()

    report = {
        "source": str(args.source),
        "avg_cpu_fps": round(meter.mean, 1),
        "athletes": [
            {"track_id": a.tid, "exercise": a.exercise, "reps": a.reps, "full_range_reps": a.good_reps,
             "form_score": round(100 * a.good_reps / a.reps) if a.reps else None,
             "issues": {k: v for k, v in a.issues.items() if not k.startswith("Good")}}
            for a in athletes.values() if a.exercise and a.frames > 15
        ],
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
