"""Human-robot collaboration safety monitor (speed & separation monitoring, ISO/TS 15066-inspired).

* People: YOLO26-pose keypoints (hands, feet, posture)
* Robots: YOLOE-26 open-vocabulary segmentation from text prompts - no robot dataset needed
* Separation: distance from any body keypoint to the robot mask, converted to metres using the
  person's own torso length as a scale reference (monocular approximation)
* Output: SAFE / SLOW / STOP state with hysteresis, recommended robot speed, event log, person-down alarm

DISCLAIMER: decision-support / analytics demo. Not a certified safety function (use PLd/SIL2 rated
scanners or safety-rated vision for actual machine guarding).

Usage:
    python run.py --source data/ur10_safety_zone.mp4 --cell ur10_cell.yaml
    python run.py --source data/cobot_assembly_line.mp4 --cell cobot_line.yaml
"""
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from config import DEVICE, MODELS, model_path  # noqa: E402
from draw import BLUE, FONT, GREEN, GREY, RED, WHITE, YELLOW, banner, label, mask_overlay, poly, watermark, with_sidebar  # noqa: E402
from video_io import FPSMeter, VideoSink, VideoSource  # noqa: E402
from zones import point_in_poly  # noqa: E402

from ultralytics import YOLO, YOLOE  # noqa: E402

STATE_COLOR = {"SAFE": GREEN, "SLOW": YELLOW, "STOP": RED}
RANK = {"SAFE": 0, "SLOW": 1, "STOP": 2}
L_SH, R_SH, L_WR, R_WR, L_HIP, R_HIP, L_AN, R_AN = 5, 6, 9, 10, 11, 12, 15, 16
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


class StateMachine:
    """Escalate immediately, de-escalate only after the lower state has held for `release` seconds."""

    def __init__(self, release=1.0):
        self.state, self.release, self.lower_since = "SAFE", release, None

    def update(self, proposed, t):
        if RANK[proposed] >= RANK[self.state]:
            changed = proposed != self.state
            self.state, self.lower_since = proposed, None
            return changed
        if self.lower_since is None:
            self.lower_since = t
        if t - self.lower_since >= self.release:
            self.state, self.lower_since = proposed, None
            return True
        return False


def person_scale(k, c, min_conf=0.4):
    """Pixels per metre at the person's depth from torso length (shoulder-mid to hip-mid ~ 0.5 m)."""
    if min(c[L_SH], c[R_SH], c[L_HIP], c[R_HIP]) < min_conf:
        return None
    torso = np.linalg.norm((k[L_SH] + k[R_SH]) / 2 - (k[L_HIP] + k[R_HIP]) / 2)
    return torso / 0.5 if torso > 10 else None


def is_down(box, k, c):
    """Person lying on the floor: wide box and horizontal torso."""
    w, h = box[2] - box[0], box[3] - box[1]
    if w < 1.3 * h:
        return False
    if min(c[L_SH], c[L_HIP]) > 0.4:
        v = k[L_SH] - k[L_HIP]
        return abs(v[0]) > 1.5 * abs(v[1])
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--cell", required=True, help="work-cell yaml (prompts, thresholds, floor zones)")
    ap.add_argument("--pose-model", default="yolo26n-pose.pt")
    ap.add_argument("--robot-model", default="yoloe-26s-seg.pt")
    ap.add_argument("--robot-every", type=int, default=3, help="run robot segmentation every N frames")
    ap.add_argument("--robot-memory", type=float, default=6.0, help="seconds to keep the last robot mask when occluded (fail-safe)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--output", default=None)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    cell = yaml.safe_load(Path(args.cell).read_text())
    zones = {k: np.float32(v) for k, v in (cell.get("floor_zones") or {}).items()}
    pose = YOLO(model_path(args.pose_model))
    robot = YOLOE(model_path(args.robot_model))
    cwd = os.getcwd()
    os.chdir(MODELS)  # the text encoder used by set_classes downloads into cwd - keep it with the weights
    n_robot = len(cell["robot_prompts"])
    # negative prompts: give the open-vocab model a better label for people / furniture than "robot"
    robot.set_classes(cell["robot_prompts"] + cell.get("negative_prompts", ["person", "table", "conveyor belt"]))
    os.chdir(cwd)

    src = VideoSource(args.source, max_seconds=args.max_seconds, width=args.width)
    out = Path(args.output or f"outputs/{Path(str(args.source)).stem}.mp4")
    sink = VideoSink(out, src.fps)
    sm = StateMachine()
    meter = FPSMeter()
    robot_mask, robot_boxes, dist_map = None, [], None
    robot_seen_t, robot_occluded = -1e9, False
    events, time_in = [], {"SAFE": 0.0, "SLOW": 0.0, "STOP": 0.0}
    min_sep_log = []

    for i, frame in enumerate(src):
        t = i / src.fps
        H, W = frame.shape[:2]
        if i % args.robot_every == 0:
            r = robot(frame, device=DEVICE, conf=cell.get("robot_conf", 0.25), iou=0.5, agnostic_nms=True, verbose=False, imgsz=640)[0]
            new_mask = np.zeros((H, W), np.uint8)
            new_boxes = []
            if r.masks is not None:
                for m_xy, b, rc in zip(r.masks.xy, r.boxes.xyxy.cpu().numpy(), r.boxes.cls.int().tolist()):
                    if rc >= n_robot:  # negative class (person, table, ...)
                        continue
                    # reject huge masks: open-vocab models sometimes swallow conveyors / fixtures
                    if len(m_xy) >= 3 and cv2.contourArea(m_xy.astype(np.float32)) < cell.get("max_robot_area", 0.4) * H * W:
                        cv2.fillPoly(new_mask, [m_xy.astype(np.int32)], 1)
                        new_boxes.append(b)
            if new_boxes:
                robot_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
                robot_boxes, robot_seen_t, robot_occluded = new_boxes, t, False
            elif t - robot_seen_t > args.robot_memory:  # robot really gone (camera moved / cell empty)
                robot_mask, robot_boxes, robot_occluded = None, [], False
            else:  # occluded by a person: keep last known robot position (fail-safe)
                robot_occluded = robot_mask is not None
            # distance (px) from every pixel to the nearest robot pixel
            dist_map = cv2.distanceTransform(1 - robot_mask, cv2.DIST_L2, 5) if robot_mask is not None and robot_mask.any() else None

        p = pose.track(frame, persist=True, device=DEVICE, conf=0.4, verbose=False, imgsz=640, tracker="bytetrack.yaml")[0]
        vis = frame.copy()
        for name, z in sorted(zones.items(), key=lambda kv: -RANK[kv[0]]):
            poly(vis, z, STATE_COLOR[name], alpha=0.18, thickness=2)
            label(vis, f"{name} ZONE", z[0] + [6, 26], color=STATE_COLOR[name], scale=0.5)
        if robot_mask is not None and robot_mask.any():
            mask_overlay(vis, robot_mask, BLUE, alpha=0.35)
            for b in robot_boxes[:3]:
                label(vis, "ROBOT (occluded - last known)" if robot_occluded else "ROBOT", (b[0], b[1] + 22), color=BLUE, scale=0.55)

        proposed, people, min_sep, down = "SAFE", 0, None, False
        if p.keypoints is not None and len(p.boxes):
            kps = p.keypoints.xy.cpu().numpy()
            kcs = p.keypoints.conf.cpu().numpy()
            boxes = p.boxes.xyxy.cpu().numpy()
            ids = p.boxes.id.int().cpu().tolist() if p.boxes.id is not None else list(range(len(boxes)))
            for k, c, box, tid in zip(kps, kcs, boxes, ids):
                people += 1
                pstate, reason = "SAFE", ""
                # 1) floor zones from feet (or box bottom if ankles hidden and box not clipped)
                feet = [k[j] for j in (L_AN, R_AN) if c[j] > 0.4]
                if not feet and box[3] < H - 8:
                    feet = [np.array([(box[0] + box[2]) / 2, box[3]])]
                for zname, z in zones.items():
                    if any(point_in_poly(f, z) for f in feet) and RANK[zname] > RANK[pstate]:
                        pstate, reason = zname, f"in {zname.lower()} zone"
                # 2) separation distance from any confident keypoint to the robot mask
                sep_m = None
                if dist_map is not None:
                    pts = [k[j] for j in range(5, 17) if c[j] > 0.4]
                    if pts:
                        d_px = min(dist_map[int(np.clip(q[1], 0, H - 1)), int(np.clip(q[0], 0, W - 1))] for q in pts)
                        scale = person_scale(k, c) or (box[3] - box[1]) / 1.7
                        sep_m = float(d_px / max(scale, 1e-3))
                        min_sep = sep_m if min_sep is None else min(min_sep, sep_m)
                        if sep_m < cell["stop_distance_m"] and RANK["STOP"] > RANK[pstate]:
                            pstate, reason = "STOP", f"{sep_m:.1f} m from robot"
                        elif sep_m < cell["slow_distance_m"] and RANK["SLOW"] > RANK[pstate]:
                            pstate, reason = "SLOW", f"{sep_m:.1f} m from robot"
                if is_down(box, k, c):
                    down, pstate, reason = True, "STOP", "PERSON DOWN"
                if RANK[pstate] > RANK[proposed]:
                    proposed = pstate
                col = STATE_COLOR[pstate]
                for a, b in SKELETON:
                    if c[a] > 0.4 and c[b] > 0.4:
                        cv2.line(vis, tuple(k[a].astype(int)), tuple(k[b].astype(int)), col, 3, cv2.LINE_AA)
                for j in (L_WR, R_WR):
                    if c[j] > 0.4:
                        cv2.circle(vis, tuple(k[j].astype(int)), 9, col, 2, cv2.LINE_AA)
                cv2.rectangle(vis, tuple(box[:2].astype(int)), tuple(box[2:].astype(int)), col, 2)
                txt = f"worker #{tid}" + (f" | {sep_m:.1f} m" if sep_m is not None else "")
                label(vis, txt, (box[0], box[1] + 24 if box[1] < 30 else box[1] - 4), color=col, scale=0.6)
                if reason and pstate != "SAFE":
                    label(vis, reason, (box[0], box[1] + 52 if box[1] < 30 else box[1] + 24), color=col, scale=0.55)

        prev = sm.state
        if sm.update(proposed, t):
            events.append({"t": round(t, 2), "from": prev, "to": sm.state, "people": people,
                           "min_separation_m": None if min_sep is None else round(min_sep, 2)})
        time_in[sm.state] += 1 / src.fps
        if min_sep is not None:
            min_sep_log.append(min_sep)
        if sm.state == "STOP":
            banner(vis, "PERSON DOWN - EMERGENCY" if down else "PROTECTIVE STOP - ROBOT HALTED", RED)
        elif sm.state == "SLOW":
            banner(vis, "REDUCED SPEED - HUMAN NEARBY", (0, 160, 230))

        fps = meter.tick()
        watermark(vis, "YOLO26-pose + YOLOE-26 open-vocab | CPU")
        canvas = with_sidebar(vis, width=330)
        x0, y = W + 20, 38
        cv2.putText(canvas, "HRC SAFETY MONITOR", (x0, y), FONT, 0.75, BLUE, 1, cv2.LINE_AA)
        y += 50
        st = sm.state
        cv2.rectangle(canvas, (x0, y - 30), (x0 + 290, y + 20), STATE_COLOR[st], -1)
        cv2.putText(canvas, st, (x0 + 14, y + 8), FONT, 1.1, WHITE, 2, cv2.LINE_AA)
        y += 54
        cv2.putText(canvas, f"Robot speed cmd: {cell['robot_speed_pct'][st]}%", (x0, y), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
        y += 28
        cv2.putText(canvas, f"Workers in view: {people}", (x0, y), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
        y += 28
        cv2.putText(canvas, f"Robots tracked: {len(robot_boxes)}" + (" (occluded)" if robot_occluded else ""), (x0, y), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
        y += 28
        sep_txt = f"{min_sep:.2f} m" if min_sep is not None else "--"
        cv2.putText(canvas, f"Min separation: {sep_txt}", (x0, y), FONT, 0.55, WHITE, 1, cv2.LINE_AA)
        y += 22
        cv2.putText(canvas, f"(stop < {cell['stop_distance_m']} m, slow < {cell['slow_distance_m']} m)", (x0, y), FONT, 0.4, GREY, 1, cv2.LINE_AA)
        y += 36
        cv2.putText(canvas, "Time in state", (x0, y), FONT, 0.5, GREY, 1, cv2.LINE_AA)
        y += 8
        total = max(sum(time_in.values()), 1e-6)
        xx = x0
        for s_name in ("SAFE", "SLOW", "STOP"):
            w = int(290 * time_in[s_name] / total)
            cv2.rectangle(canvas, (xx, y), (xx + w, y + 14), STATE_COLOR[s_name], -1)
            xx += w
        y += 34
        for s_name in ("SAFE", "SLOW", "STOP"):
            cv2.putText(canvas, f"{s_name:<5} {time_in[s_name]:5.1f}s", (x0, y), FONT, 0.45, STATE_COLOR[s_name], 1, cv2.LINE_AA)
            y += 20
        y += 16
        cv2.putText(canvas, "Event log", (x0, y), FONT, 0.5, GREY, 1, cv2.LINE_AA)
        y += 22
        for e in events[-6:][::-1]:
            cv2.putText(canvas, f"{e['t']:5.1f}s  {e['from']} -> {e['to']}", (x0, y), FONT, 0.45, STATE_COLOR[e["to"]], 1, cv2.LINE_AA)
            y += 20
        cv2.putText(canvas, f"CPU {fps:4.1f} FPS", (x0, canvas.shape[0] - 18), FONT, 0.42, GREY, 1, cv2.LINE_AA)
        sink.write(canvas)
        if args.show:
            cv2.imshow("hrc", canvas)
            if cv2.waitKey(1) == 27:
                break
    sink.close()

    summary = {"source": str(args.source), "avg_cpu_fps": round(meter.mean, 1),
               "time_in_state_s": {k: round(v, 1) for k, v in time_in.items()},
               "min_separation_m": round(min(min_sep_log), 2) if min_sep_log else None,
               "stops": sum(1 for e in events if e["to"] == "STOP"), "events": events}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    # Each event is what would be published to the cell PLC / MES (e.g. OPC UA node or MQTT topic cell/<id>/safety)
    print(json.dumps({k: v for k, v in summary.items() if k != "events"}, indent=2))


if __name__ == "__main__":
    main()
