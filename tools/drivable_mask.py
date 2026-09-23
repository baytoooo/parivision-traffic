"""Rebuild src/parivision/assets/drivable.png: where moving vehicles were seen in the sample clips.

    python tools/drivable_mask.py [--check]

Foot points of vehicle tracks moving faster than 40 px/s (reference view, all
four clips, from cache/tracks), blurred, thresholded and closed. The jaywalking
rule only looks for pedestrians inside this area (plus the junction box), so a
person standing at a kerb or on the median edge is not on the carriageway.
--check compares the result with the committed file instead of overwriting it.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parivision import scene as S  # noqa: E402

CLIPS = ["C3896", "C3897", "C3902", "C3905"]
MIN_SPEED = 40.0   # px/s: standing and parked vehicles do not make an area drivable
SIGMA = 5.0        # px
THRESHOLD = 0.02   # blurred hits per pixel
CLOSE = 25         # px, fills gaps between lanes


def build() -> np.ndarray:
    w, h = S.REF_SIZE
    hits = np.zeros((h, w), np.float32)
    for clip in CLIPS:
        for tr in pickle.load(open(ROOT / "cache/tracks" / f"{clip}.pkl", "rb"))["trajectories"]:
            if not tr.is_vehicle:
                continue
            p = np.round(tr.foot[tr.speed > MIN_SPEED]).astype(int)
            ok = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
            np.add.at(hits, (p[ok, 1], p[ok, 0]), 1)
    mask = (cv2.GaussianBlur(hits, (0, 0), SIGMA) > THRESHOLD).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((CLOSE, CLOSE), np.uint8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    mask = build()
    if args.check:
        old = cv2.imread(str(S.DRIVABLE), cv2.IMREAD_GRAYSCALE) > 127
        new = mask > 0
        print(f"IoU with the committed mask: {(old & new).sum() / max(1, (old | new).sum()):.4f}")
        return
    cv2.imwrite(str(S.DRIVABLE), mask * 255)
    print("wrote", S.DRIVABLE)


if __name__ == "__main__":
    main()
