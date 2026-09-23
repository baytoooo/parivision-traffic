"""Vehicle signal phase from the three-lamp head on the median nose.

Each lamp gets a small patch around its centre (mapped from reference pixels
into the frame). A lit red lamp is much redder than its dark housing, a lit
green one much greener, and a lit yellow one much warmer. Each lamp's contrast
is divided by its own threshold (``MIN_CONTRAST``) and the lamp furthest above
it wins; when every lamp is below its threshold the head is "off"
(the flashing-green part of the cycle, or a car hiding the head). Lamp
position, not just colour, decides the phase, so tail lights passing behind
the head do not read as red.

Sequence at this junction: green (the last 3 s flashing), 3 s yellow, red,
then back to green. The afternoon plan lights red and yellow together for the
last 3 s of red.
"""
from __future__ import annotations

import cv2
import numpy as np

from .scene import SIGNAL_LAMPS

RED, YELLOW, GREEN, OFF, UNKNOWN = "red", "yellow", "green", "off", "unknown"
LAMPS = ("red", "yellow", "green")
_CODES = {RED: 0, YELLOW: 1, GREEN: 2, OFF: 3, UNKNOWN: 4}
_NAMES = {v: k for k, v in _CODES.items()}
# Colour contrast (0-255 scale) each lamp needs to count as lit. The green LED
# looks cyan in sunlight, so its green-minus-red/blue contrast is much lower.
MIN_CONTRAST = np.array([20.0, 20.0, 6.0], np.float32)


def lamp_patches(H_ref_to_frame: np.ndarray, radius_ref: float = 2.5) -> list[tuple[int, int, int, int]]:
    """Frame-pixel boxes around the red, yellow and green lamp centres."""
    boxes = []
    scale = float(np.sqrt(abs(np.linalg.det(H_ref_to_frame[:2, :2]))))
    r = max(1.0, radius_ref * scale)
    for name in LAMPS:
        x, y = SIGNAL_LAMPS[name]
        p = cv2.perspectiveTransform(np.float64([[[x, y]]]), H_ref_to_frame)[0, 0]
        boxes.append((int(round(p[0] - r)), int(round(p[1] - r)), int(round(p[0] + r)) + 1, int(round(p[1] + r)) + 1))
    return boxes


def lamp_scores(frame: np.ndarray, boxes) -> np.ndarray:
    """Colour contrast of each lamp: [redness of top, warmth of middle, greenness of bottom]."""
    out = np.zeros(3, np.float32)
    h, w = frame.shape[:2]
    for k, (x0, y0, x1, y1) in enumerate(boxes):
        patch = frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1)].reshape(-1, 3).astype(np.float32)
        if len(patch) == 0:
            continue
        b, g, r = patch[:, 0], patch[:, 1], patch[:, 2]
        if k == 0:
            v = r - np.maximum(g, b)
        elif k == 1:
            v = r - b
        else:
            v = g - np.maximum(r, b)
        out[k] = float(np.percentile(v, 75))
    return out


def phase_from_scores(scores: np.ndarray, min_contrast: np.ndarray = MIN_CONTRAST) -> str:
    ratio = np.asarray(scores, np.float32) / min_contrast
    k = int(np.argmax(ratio))
    return LAMPS[k] if ratio[k] >= 1.0 else OFF


def fill_phases(raw: list[str], times: np.ndarray, max_off: float = 4.0) -> list[str]:
    """Offline clean-up: 1-sample blips removed, short dark spells take the phase before them.

    Flashing green shows up as green/off alternation, so it stays green; a dark
    spell longer than ``max_off`` becomes unknown.
    """
    codes = [_CODES[p] for p in raw]
    n = len(codes)
    for i in range(1, n - 1):  # isolated single-sample flips
        if codes[i - 1] == codes[i + 1] != codes[i] and codes[i - 1] != _CODES[OFF]:
            codes[i] = codes[i - 1]
    out = list(codes)
    i = 0
    while i < n:
        if codes[i] != _CODES[OFF]:
            i += 1
            continue
        j = i
        while j < n and codes[j] == _CODES[OFF]:
            j += 1
        prev = codes[i - 1] if i > 0 else (codes[j] if j < n else _CODES[UNKNOWN])
        span = (times[j - 1] - times[i]) if j > i else 0.0
        fill = prev if span <= max_off else _CODES[UNKNOWN]
        for k in range(i, j):
            out[k] = fill
        i = j
    return [_NAMES[c] for c in out]
