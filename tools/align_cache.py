"""Per-clip homography to the reference view, from a median background of the proxy.

    python tools/align_cache.py C3896 C3897 C3902 C3905
Writes cache/align/<clip>.npz with H (1920-wide working pixels -> reference pixels),
the inlier count and the background warped into the reference view.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parivision.registration import align, background  # noqa: E402
from parivision.scene import REF_SIZE  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--every", type=float, default=15.0)
    args = ap.parse_args()
    out = ROOT / "cache/align"
    out.mkdir(parents=True, exist_ok=True)
    refs = [cv2.imread(str(p)) for p in sorted((ROOT / "src/parivision/assets").glob("reference*.jpg"))]
    for clip in args.clips:
        cap = cv2.VideoCapture(str(ROOT / "cache/proxy" / f"{clip}.mp4"))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for idx in range(0, n, int(args.every * fps)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, f = cap.read()
            if ok:
                frames.append(cv2.resize(f, REF_SIZE, interpolation=cv2.INTER_CUBIC))
        bg = background(frames)
        best = max((align(bg, r) for r in refs), key=lambda a: a.inliers)
        warped = cv2.warpPerspective(bg, best.H, REF_SIZE)
        np.savez(out / f"{clip}.npz", H=best.H, inliers=best.inliers)
        cv2.imwrite(str(out / f"{clip}_bg_ref.jpg"), warped)
        print(clip, "inliers", best.inliers, "H", np.round(best.H, 4).tolist())


if __name__ == "__main__":
    main()
