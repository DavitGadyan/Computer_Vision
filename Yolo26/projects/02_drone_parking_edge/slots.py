"""Build parking-slot polygons from a parametric row config, or load hand-annotated polygons."""
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "common"))
from zones import load_polygons  # noqa: E402


def slots_from_rows(cfg_path, frame_width, inset=0.08):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    s = frame_width / cfg.get("reference_width", frame_width)
    slots = {}
    for row, r in cfg["rows"].items():
        for k in range(r["count"]):
            xa, xb = r["x0"] + k * r["pitch"], r["x0"] + (k + 1) * r["pitch"]
            quad = np.array([[xa, r["y_top"]], [xb, r["y_top"]], [xb + r["dx"], r["y_bottom"]], [xa + r["dx"], r["y_bottom"]]], np.float32)
            c = quad.mean(0)
            quad = c + (quad - c) * (1 - inset)  # small inset so neighbours' bumpers don't count
            slots[f"{row}{k + 1:02d}"] = quad * s
    return slots


def load_slots(path, frame_width):
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        return slots_from_rows(path, frame_width)
    return load_polygons(path)
