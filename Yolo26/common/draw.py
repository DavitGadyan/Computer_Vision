"""Consistent visual language for all demo overlays (HUD panels, labels, zones)."""
import cv2
import numpy as np

# BGR palette
GREEN = (80, 200, 60)
YELLOW = (0, 210, 255)
RED = (60, 60, 230)
BLUE = (230, 160, 40)
WHITE = (255, 255, 255)
DARK = (25, 25, 25)
GREY = (160, 160, 160)
FONT = cv2.FONT_HERSHEY_DUPLEX


def color_for_id(i: int):
    """Stable distinct colour per track id."""
    rng = np.random.default_rng(int(i) * 7919 + 13)
    return tuple(int(c) for c in rng.integers(70, 255, 3))


def label(img, text, org, color=GREEN, scale=0.5, thickness=1, text_color=WHITE, pad=4):
    """Filled label box with text; org is the bottom-left anchor of the box."""
    (w, h), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = int(org[0]), int(org[1])
    y = max(y, h + 2 * pad)
    cv2.rectangle(img, (x, y - h - 2 * pad), (x + w + 2 * pad, y + base - pad // 2), color, -1)
    cv2.putText(img, text, (x + pad, y - pad), FONT, scale, text_color, thickness, cv2.LINE_AA)


def panel(img, lines, org=(12, 12), title=None, width=None, alpha=0.65, scale=0.55, accent=BLUE):
    """Semi-transparent HUD panel. lines: list of str or (str, color)."""
    lines = [(l, WHITE) if isinstance(l, str) else l for l in lines]
    lh = int(26 * scale / 0.55)
    w = width or max([cv2.getTextSize(t, FONT, scale, 1)[0][0] for t, _ in lines] + [180]) + 24
    h = lh * len(lines) + 16 + (lh + 6 if title else 0)
    x, y = org
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), DARK, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + 4, y + h), accent, -1)
    cy = y + 8
    if title:
        cv2.putText(img, title, (x + 14, cy + lh - 6), FONT, scale * 1.05, accent, 1, cv2.LINE_AA)
        cy += lh + 6
    for text, col in lines:
        cv2.putText(img, text, (x + 14, cy + lh - 8), FONT, scale, col, 1, cv2.LINE_AA)
        cy += lh
    return x + w, y + h


def poly(img, pts, color, alpha=0.25, thickness=2):
    pts = np.asarray(pts, np.int32)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)


def mask_overlay(img, mask, color, alpha=0.45):
    """Blend a boolean mask onto img in-place."""
    m = mask.astype(bool)
    img[m] = (img[m] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)


def watermark(img, text="YOLO26 | CPU inference"):
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.45, 1)
    cv2.putText(img, text, (w - tw - 12, h - 12), FONT, 0.45, (220, 220, 220), 1, cv2.LINE_AA)


def banner(img, text, color=RED, scale=0.9):
    """Centered top alert banner."""
    h, w = img.shape[:2]
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 2)
    x = (w - tw) // 2
    cv2.rectangle(img, (x - 16, 8), (x + tw + 16, 8 + th + 20), color, -1)
    cv2.putText(img, text, (x, 8 + th + 10), FONT, scale, WHITE, 2, cv2.LINE_AA)


def with_sidebar(img, width=330, color=(32, 32, 32)):
    """Return a canvas with a dark dashboard column on the right; widgets are drawn at x >= img width."""
    h, w = img.shape[:2]
    canvas = np.full((h, w + width, 3), color, np.uint8)
    canvas[:, :w] = img
    cv2.line(canvas, (w, 0), (w, h), (70, 70, 70), 2)
    return canvas


def bar(img, org, size, frac, color, bg=(60, 60, 60)):
    x, y = org
    w, h = size
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    cv2.rectangle(img, (x, y), (x + int(w * max(0.0, min(1.0, frac))), y + h), color, -1)
