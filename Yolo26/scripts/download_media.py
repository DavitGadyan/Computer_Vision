"""Download the reference clips used by each demo into projects/<project>/data/ (gitignored).

Clips are short excerpts of publicly available YouTube videos, used for non-commercial
technical demonstration only. Credit / source links are listed in each project README.
"""
import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# project, output name, youtube id, start (s), duration (s), description[, max height]
CLIPS = [
    ("01_workout_monitoring", "pushups.mp4", "TIHiJS_VV4Y", 0, 16, "Man doing push-ups (free-to-use clip)"),
    ("01_workout_monitoring", "squats_gym.mp4", "8uoaYwS6iFM", 20, 14, "IU Health - bodyweight squat coaching"),
    ("01_workout_monitoring", "squats_home.mp4", "gmVDpETAEE4", 0, 13, "Squats stock video"),
    ("02_drone_parking_edge", "parking_drone.mp4", "FO8IF1Swcv4", 0, 35, "Top-view drone footage of a parking lot"),
    ("03_solar_panel_inspection", "el_cracked_module.mp4", "wL2ZlHTXRcw", 0, 60, "Electroluminescence of PV module (2 Degrees Kelvin)"),
    ("04_traffic_segmentation_speed_anpr", "traffic_overpass.mp4", "FsGPxhidwGg", 0, 60, "Overpass traffic camera - plate detection test", 1080),
    ("05_robot_cell_safety", "ur10_safety_zone.mp4", "-9-KgZXAPq0", 7, 40, "Safety zone demo, UR10 (Axis Systems)"),
    ("05_robot_cell_safety", "cobot_assembly_line.mp4", "u-LbWc0TQcU", 0, 28, "Cobots on appliance assembly line (EVS)"),
]


def download(project, name, vid, start, dur, desc="", height=720, force=False):
    out = ROOT / "projects" / project / "data" / name
    if out.exists() and not force:
        print("exists", out.relative_to(ROOT))
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        subprocess.run(["yt-dlp", "-q", "--no-warnings", "-f", f"bv*[height<={height}][vcodec^=avc1]/bv*[height<={height}]",
                        "--download-sections", f"*{start}-{start + dur}", "--force-keyframes-at-cuts",
                        "-o", str(raw), f"https://youtu.be/{vid}"], check=True)
        # normalise to H.264 / no audio so OpenCV can always decode it
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-t", str(dur), "-an", "-c:v", "libx264",
                        "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out)], check=True)
    print("saved", out.relative_to(ROOT))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", help="only download clips for this project folder name")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not shutil.which("yt-dlp") or not shutil.which("ffmpeg"):
        raise SystemExit("yt-dlp and ffmpeg are required")
    for c in CLIPS:
        if args.project is None or args.project == c[0]:
            download(*c, force=args.force)


if __name__ == "__main__":
    main()
