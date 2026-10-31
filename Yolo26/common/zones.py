"""Polygon zone utilities shared by parking, traffic and robot-safety projects."""
import json
from pathlib import Path

import cv2
import numpy as np


def load_polygons(path):
    """JSON format: {"<name>": [[x, y], ...], ...} in pixel coordinates of the reference frame."""
    return {k: np.asarray(v, np.float32) for k, v in json.loads(Path(path).read_text()).items()}


def save_polygons(path, polys):
    Path(path).write_text(json.dumps({k: np.asarray(v).round().astype(int).tolist() for k, v in polys.items()}, indent=1))


def point_in_poly(pt, poly) -> bool:
    return cv2.pointPolygonTest(np.asarray(poly, np.float32), (float(pt[0]), float(pt[1])), False) >= 0


def poly_iou(a, b) -> float:
    """IoU of two convex polygons (OBB vs parking slot)."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0


def poly_coverage(slot, box) -> float:
    """Fraction of the slot area covered by the box polygon (robust to cars larger than slots)."""
    slot, box = np.asarray(slot, np.float32), np.asarray(box, np.float32)
    inter, _ = cv2.intersectConvexConvex(slot, box)
    a = cv2.contourArea(slot)
    return float(inter / a) if a > 0 else 0.0


def scale_polys(polys, sx, sy):
    return {k: v * np.array([sx, sy], np.float32) for k, v in polys.items()}
