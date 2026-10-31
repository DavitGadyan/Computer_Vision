"""Shared paths and device policy. Everything runs on CPU so the GPU stays free for other workloads."""
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # hide GPU from torch before it is imported

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
DEVICE = "cpu"
MODELS.mkdir(exist_ok=True)


def model_path(name: str) -> str:
    """Resolve a weight file inside ./models (Ultralytics auto-downloads official weights there)."""
    p = Path(name)
    if p.exists():
        return str(p)
    return str(MODELS / p.name)
