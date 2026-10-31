"""Turn annotated demo outputs into portfolio media.

For one project it writes
  projects/<p>/assets/demo.gif, demo.jpg           (README preview)
  upwork_portfolio/<p>/demo.mp4                    (H.264, <10 MB, for Upwork / LinkedIn upload)
  upwork_portfolio/<p>/cover.png                   (1600x1200, 4:3 portfolio cover with title)
  upwork_portfolio/<p>/screenshot_<n>.jpg          (full-res stills)

Usage:
    python scripts/make_assets.py --project 01_workout_monitoring --title "AI Workout Monitor" \
        --subtitle "Rep counting & form feedback" \
        --seg projects/01_workout_monitoring/outputs/pushups.mp4:0:8 --seg ...:0:7 --stills 3,12
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FONT = cv2.FONT_HERSHEY_DUPLEX


def run(cmd):
    subprocess.run(cmd, check=True)


def build_montage(segs, out, width=1280, crf=27):
    """Concatenate (path, start, dur) segments, scaled to a common width, into one H.264 mp4."""
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for k, (p, s, d) in enumerate(segs):
            part = Path(tmp) / f"p{k}.mp4"
            run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(s), "-t", str(d), "-i", str(p), "-an",
                 "-vf", f"scale={width}:-2:flags=lanczos,fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "18", "-preset", "fast", str(part)])
            parts.append(part)
        # pad parts to the same height (different aspect ratios) then concat
        hs = [int(cv2.VideoCapture(str(p)).get(cv2.CAP_PROP_FRAME_HEIGHT)) for p in parts]
        H = max(hs)
        lst = Path(tmp) / "list.txt"
        padded = []
        for k, (p, h) in enumerate(zip(parts, hs)):
            if h != H:
                q = Path(tmp) / f"q{k}.mp4"
                run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(p), "-vf", f"pad={width}:{H}:0:(oh-ih)/2:color=0x202020",
                     "-c:v", "libx264", "-crf", "18", "-preset", "fast", str(q)])
                p = q
            padded.append(p)
        lst.write_text("".join(f"file '{p}'\n" for p in padded))
        for c in (crf, crf + 3, crf + 6):  # keep under ~9.5 MB for upload limits
            run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:v", "libx264",
                 "-crf", str(c), "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)])
            if out.stat().st_size < 9.5e6:
                break


def make_gif(mp4, gif, start, dur, width=720, fps=10):
    vf = f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
    run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-t", str(dur), "-i", str(mp4), "-vf", vf, str(gif)])
    if gif.stat().st_size > 6e6 and width > 480:  # shrink if too heavy for a README
        make_gif(mp4, gif, start, dur, width=width - 120, fps=fps)


def grab(mp4, t):
    cap = cv2.VideoCapture(str(mp4))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, f = cap.read()
    return f if ok else None


def cover(frame, title, subtitle, tags, out, size=(1600, 1200)):
    W, H = size
    canvas = np.full((H, W, 3), (28, 24, 22), np.uint8)
    band = 250
    fh, fw = frame.shape[:2]
    s = min(W / fw, (H - band) / fh)
    img = cv2.resize(frame, (int(fw * s), int(fh * s)), interpolation=cv2.INTER_AREA)
    x = (W - img.shape[1]) // 2
    canvas[band:band + img.shape[0], x:x + img.shape[1]] = img
    cv2.rectangle(canvas, (0, 0), (W, band), (40, 32, 26), -1)
    cv2.rectangle(canvas, (0, band - 8), (W, band), (230, 160, 40), -1)
    cv2.putText(canvas, title, (60, 110), FONT, 2.2, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (62, 170), FONT, 1.05, (230, 190, 120), 2, cv2.LINE_AA)
    tx = 62
    for t in tags:
        (tw, th), _ = cv2.getTextSize(t, FONT, 0.75, 1)
        cv2.rectangle(canvas, (tx, 192), (tx + tw + 24, 192 + th + 18), (230, 160, 40), -1)
        cv2.putText(canvas, t, (tx + 12, 192 + th + 8), FONT, 0.75, (25, 25, 25), 1, cv2.LINE_AA)
        tx += tw + 40
    # fill the area below the image (if any) with a subtle footer
    bottom = band + img.shape[0]
    if H - bottom > 40:
        cv2.putText(canvas, "Ultralytics YOLO26  |  production-ready computer vision", (62, H - 30), FONT, 0.8, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), canvas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--subtitle", required=True)
    ap.add_argument("--tags", default="YOLO26,Python,OpenCV")
    ap.add_argument("--seg", action="append", required=True, help="path:start:duration (repeatable)")
    ap.add_argument("--gif", default="0:8", help="start:duration inside the montage")
    ap.add_argument("--stills", default="2", help="comma-separated montage timestamps for screenshots (first = cover)")
    args = ap.parse_args()

    assets = ROOT / "projects" / args.project / "assets"
    up = ROOT / "upwork_portfolio" / args.project
    assets.mkdir(parents=True, exist_ok=True)
    up.mkdir(parents=True, exist_ok=True)
    segs = []
    for s in args.seg:
        p, st, d = s.rsplit(":", 2)
        segs.append((ROOT / p if not Path(p).is_absolute() else Path(p), float(st), float(d)))

    mp4 = up / "demo.mp4"
    build_montage(segs, mp4)
    gs, gd = map(float, args.gif.split(":"))
    make_gif(mp4, assets / "demo.gif", gs, gd)
    stills = [float(t) for t in args.stills.split(",")]
    for k, t in enumerate(stills):
        f = grab(mp4, t)
        if f is None:
            continue
        cv2.imwrite(str(up / f"screenshot_{k + 1}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if k == 0:
            cv2.imwrite(str(assets / "demo.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 88])
            cover(f, args.title, args.subtitle, args.tags.split(","), up / "cover.png")
    for p in [mp4, assets / "demo.gif", up / "cover.png"]:
        print(f"{p.relative_to(ROOT)}  {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
