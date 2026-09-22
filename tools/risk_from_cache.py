"""Replay cached detections through the Part B hazard model (calibration without decoding video).

    python tools/risk_from_cache.py C3896 C3897 C3902 C3905
Prints the score distribution and the alarms (runs of score >= 0.5) per clip.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parivision.detector import Detections  # noqa: E402
from parivision.registration import Alignment  # noqa: E402
from parivision.risk import Anticipator  # noqa: E402


def replay(clip: str) -> tuple[np.ndarray, np.ndarray]:
    npz = next((ROOT / "cache/det").glob(f"{clip}__*.npz"))
    z = np.load(npz)
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


if __name__ == "__main__":
    out = {}
    for clip in sys.argv[1:]:
        ts, sc = replay(clip)
        al = alarms(ts, sc)
        out[clip] = {"t": ts.round(2).tolist(), "risk": sc.round(4).tolist()}
        print(f"{clip}: p50={np.median(sc):.3f} p99={np.percentile(sc, 99):.3f} max={sc.max():.3f} "
              f"alarms={len(al)} {[[round(a, 1), round(b, 1)] for a, b in al][:10]}")
    (ROOT / "cache/risk_replay.json").write_text(json.dumps(out))
