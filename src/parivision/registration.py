"""Align a video to the reference view of the intersection.

The sample clips are shot from the same spot but not with identical framing
(zoom and pan differ between clips), so every piece of scene geometry lives in
one reference image and is mapped into each video with a homography estimated
from static background features.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Alignment:
    H: np.ndarray  # 3x3, maps video pixels (working resolution) -> reference pixels
    inliers: int
    ok: bool


def warp_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    return cv2.perspectiveTransform(pts[None], H)[0]


def background(frames: list[np.ndarray]) -> np.ndarray:
    """Per-pixel median of a handful of frames: removes most moving traffic."""
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _features(img: np.ndarray):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    sift = cv2.SIFT_create(nfeatures=6000)
    return sift.detectAndCompute(clahe.apply(gray), None)


def align(frame: np.ndarray, reference: np.ndarray, min_inliers: int = 60, work_width: int = 960) -> Alignment:
    """Homography from ``frame`` to ``reference`` (both BGR, any size).

    Features are matched on ``work_width``-wide copies (4x cheaper than 1920 and
    just as accurate for a whole-frame homography); the result is rescaled.
    """
    sf = work_width / frame.shape[1]
    sr = work_width / reference.shape[1]
    H = _align_small(cv2.resize(frame, None, fx=sf, fy=sf, interpolation=cv2.INTER_AREA),
                     cv2.resize(reference, None, fx=sr, fy=sr, interpolation=cv2.INTER_AREA), min_inliers)
    if H is None:
        return _fallback(frame, reference)
    H_full = np.diag([1 / sr, 1 / sr, 1.0]) @ H[0] @ np.diag([sf, sf, 1.0])
    return Alignment(H_full / H_full[2, 2], H[1], True)


_REF_FEATURES: dict[tuple, tuple] = {}


def _reference_features(reference: np.ndarray):
    """The references are the same few images all run long: compute their features once."""
    key = (reference.shape, reference.tobytes()[::4099])
    if key not in _REF_FEATURES:
        _REF_FEATURES[key] = _features(reference)
    return _REF_FEATURES[key]


def _align_small(frame: np.ndarray, reference: np.ndarray, min_inliers: int):
    k1, d1 = _features(frame)
    k2, d2 = _reference_features(reference)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None
    matcher = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 5}, {"checks": 64})
    pairs = matcher.knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < min_inliers:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0, maxIters=5000, confidence=0.999)
    inliers = int(mask.sum()) if mask is not None else 0
    if H is None or inliers < min_inliers:
        return None
    return H, inliers


def align_best(frame: np.ndarray, references: list[np.ndarray]) -> Alignment:
    """Align against every reference view (day, dusk) and keep the best-supported homography.

    All references are already expressed in reference-view pixels, so any of
    them gives a homography into the same coordinate frame.
    """
    return align_best_with(frame, references)[0]


def align_best_with(frame: np.ndarray, references: list[np.ndarray]) -> tuple[Alignment, np.ndarray]:
    """align_best, and the reference that won (later frames are registered against it alone)."""
    results = [(align(frame, ref), ref) for ref in references]
    return max(results, key=lambda r: (r[0].ok, r[0].inliers))


def _fallback(frame: np.ndarray, reference: np.ndarray) -> Alignment:
    """Plain rescale: right when the clip has the reference framing, the best guess otherwise."""
    sx = reference.shape[1] / frame.shape[1]
    sy = reference.shape[0] / frame.shape[0]
    return Alignment(np.diag([sx, sy, 1.0]), 0, False)
