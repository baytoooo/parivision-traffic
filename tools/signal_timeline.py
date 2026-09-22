"""Vehicle-signal phase timeline for a clip, read from 1920-wide frames of the original video.

    python tools/signal_timeline.py C3896 C3897 C3902 C3905 --fps 5
Needs cache/align/<clip>.npz (tools/align_cache.py).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parivision.signal import fill_phases, lamp_patches, lamp_scores, phase_from_scores  # noqa: E402
from parivision.video import sample_frames  # noqa: E402


def segments(times, phases):
    segs = []
    for t, p in zip(times, phases):
        if segs and segs[-1][2] == p:
            segs[-1][1] = t
        else:
            segs.append([t, t, p])
    return [[round(a, 1), round(b, 1), p] for a, b, p in segs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--fps", type=float, default=5.0)
    args = ap.parse_args()
    out_dir = ROOT / "cache/signal"
    out_dir.mkdir(parents=True, exist_ok=True)
    for clip in args.clips:
        H = np.load(ROOT / "cache/align" / f"{clip}.npz")["H"]
        boxes = lamp_patches(np.linalg.inv(H))
        times, scores = [], []
        for _, t, img in sample_frames(ROOT / "kit/samples" / f"{clip}.MP4", args.fps, 1920):
            times.append(t)
            scores.append(lamp_scores(img, boxes).tolist())
        raw = [phase_from_scores(np.array(s)) for s in scores]
        phases = fill_phases(raw, np.array(times))
        segs = segments(times, phases)
        (out_dir / f"{clip}.json").write_text(json.dumps(
            {"segments": segs, "times": times, "phases": phases, "raw": raw, "scores": scores}))
        print(clip, " ".join(f"{a:.0f}-{b:.0f}:{p}" for a, b, p in segs), flush=True)


if __name__ == "__main__":
    main()
