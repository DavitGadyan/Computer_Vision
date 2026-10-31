"""Video reading/writing helpers. Output is re-encoded to H.264 with ffmpeg so it plays on GitHub/Upwork."""
import shutil
import subprocess
import time
from pathlib import Path

import cv2


class VideoSource:
    """Iterate frames from a file, webcam index or RTSP URL, optionally limiting duration and resizing."""

    def __init__(self, source, max_seconds=None, start_seconds=0.0, width=None):
        self.cap = cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video source: {source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if start_seconds:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)
        self.max_frames = int(max_seconds * self.fps) if max_seconds else None
        self.width = width
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        n = 0
        while self.max_frames is None or n < self.max_frames:
            ok, frame = self.cap.read()
            if not ok:
                break
            if self.width and frame.shape[1] != self.width:
                h = int(frame.shape[0] * self.width / frame.shape[1]) // 2 * 2
                frame = cv2.resize(frame, (self.width, h), interpolation=cv2.INTER_AREA)
            n += 1
            yield frame
        self.cap.release()


class VideoSink:
    """Write frames to a temporary mp4v file, then transcode to browser-friendly H.264 on close."""

    def __init__(self, path, fps, crf=26):
        self.path, self.fps, self.crf = Path(path), fps, crf
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tmp = self.path.with_suffix(".tmp.mp4")
        self.writer = None

    def write(self, frame):
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(self.tmp), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self.writer.write(frame)

    def close(self):
        if self.writer is None:
            return
        self.writer.release()
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(self.tmp), "-c:v", "libx264", "-preset", "slow",
                            "-crf", str(self.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self.path)], check=True)
            self.tmp.unlink()
        else:
            self.tmp.rename(self.path)


def export_gif(mp4, gif, width=640, fps=10, start=0, duration=8):
    """High-quality palette GIF for README previews."""
    vf = f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-t", str(duration), "-i", str(mp4), "-vf", vf, str(gif)], check=True)


def export_thumbnail(mp4, png, at=2.0, width=960):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(at), "-i", str(mp4), "-frames:v", "1",
                    "-vf", f"scale={width}:-1", str(png)], check=True)


class FPSMeter:
    """Exponential moving average of processing FPS (shown on HUD and reported in READMEs)."""

    def __init__(self, alpha=0.1):
        self.alpha, self.fps, self.t = alpha, None, None
        self.samples = []

    def tick(self):
        now = time.perf_counter()
        if self.t is not None:
            inst = 1.0 / max(now - self.t, 1e-6)
            self.fps = inst if self.fps is None else (1 - self.alpha) * self.fps + self.alpha * inst
            self.samples.append(inst)
        self.t = now
        return self.fps or 0.0

    @property
    def mean(self):
        return sum(self.samples) / len(self.samples) if self.samples else 0.0
