"""Replay cached detections through the Part B hazard model (calibration without decoding video).

    python tools/risk_from_cache.py C3896 C3897 C3902 C3905 [--tag yolo26s_960_1920_10fps]
Prints the score distribution and the alarms (runs of score >= 0.5) per clip.

Reads cache/det/<clip>__<tag>.npz (tools/cache_detections.py) and cache/align (tools/align_cache.py).
The default tag is the cache closest to the GPU Part B detector (yolo26s at 960). Only C3897 and
C3905 have it; for a clip without it we take the clip's first cache in name order and print which.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parivision.detector import Detections  # noqa: E402
from parivision.registration import Alignment  # noqa: E402
from parivision.risk import Anticipator  # noqa: E402

DEFAULT_TAG = "yolo26s_960_1920_10fps"


def cache_for(clip: str, tag: str) -> Path:
    """cache/det/<clip>__<tag>.npz, or the clip's first cache in name order when that one is missing."""
    npz = ROOT / "cache/det" / f"{clip}__{tag}.npz"
    if npz.exists():
        return npz
    found = sorted((ROOT / "cache/det").glob(f"{clip}__*.npz"))
    if not found:
        sys.exit(f"{clip}: no detector cache in cache/det (run tools/cache_detections.py first)")
    print(f"{clip}: no {npz.name}, replaying {found[0].name}")
    return found[0]


def replay(clip: str, tag: str = DEFAULT_TAG) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(cache_for(clip, tag))
    det, fps, stride = z["det"], float(z["fps"]), int(z["stride"])
    w, h = int(z["width"]), int(z["height"])
    H = np.load(ROOT / "cache/align" / f"{clip}.npz")["H"]
    model = Anticipator({"fps": fps / stride, "width": w, "height": h})
    model.alignment = Alignment(H, 1, True)
    frames = np.arange(0, int(z["n_frames"]), stride)
    by_frame = {int(f): det[det[:, 0] == f] for f in np.unique(det[:, 0])}
    ts, scores = [], []
    for f in frames:
        rows = by_frame.get(int(f))
        rows = rows[rows[:, 6] >= 0.2] if rows is not None else None
        d = Detections.empty() if rows is None or not len(rows) else Detections(
            rows[:, 2:6].astype(np.float32), rows[:, 6].astype(np.float32), rows[:, 7].astype(np.int16))
        ts.append(f / fps)
        scores.append(model.observe(d, (h, w), f / fps))
    return np.array(ts), np.array(scores)


def alarms(ts, scores, theta=0.5, gap=2.0):
    runs, start = [], None
    for t, s in zip(ts, scores):
        if s >= theta and start is None:
            start = t
        elif s < theta and start is not None:
            runs.append([start, t]); start = None
    if start is not None:
        runs.append([start, ts[-1]])
    merged = []
    for a, b in runs:
        if merged and a - merged[-1][1] < gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return merged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="detector cache to replay, cache/det/<clip>__<tag>.npz")
    args = ap.parse_args()
    for clip in args.clips:
        ts, sc = replay(clip, args.tag)
        al = alarms(ts, sc)
        print(f"{clip}: p50={np.median(sc):.3f} p99={np.percentile(sc, 99):.3f} max={sc.max():.3f} "
              f"alarms={len(al)} {[[round(a, 1), round(b, 1)] for a, b in al][:10]}")


if __name__ == "__main__":
    main()
