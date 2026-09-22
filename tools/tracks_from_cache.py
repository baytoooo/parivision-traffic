"""Track cached detections and save trajectories (reference coordinates) for rule development and EDA.

    python tools/tracks_from_cache.py cache/det/*.npz --out cache/tracks
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parivision.detector import Detections  # noqa: E402
from parivision.scene import REF_SIZE  # noqa: E402
from parivision.tracking import MultiTracker, collect  # noqa: E402
from parivision.trajectories import build  # noqa: E402


def track_cache(npz_path: Path):
    z = np.load(npz_path)
    det, fps, stride = z["det"], float(z["fps"]), int(z["stride"])
    w, h = int(z["width"]), int(z["height"])
    tracker = MultiTracker(fps=fps / stride)
    tracks: dict = {}
    frames = np.unique(det[:, 0]).astype(int)
    # frames without any detection still advance the tracker
    all_frames = np.arange(0, int(z["n_frames"]), stride)
    by_frame = {f: det[det[:, 0] == f] for f in frames}
    for f in all_frames:
        rows = by_frame.get(f)
        d = Detections.empty() if rows is None else Detections(
            rows[:, 2:6].astype(np.float32), rows[:, 6].astype(np.float32), rows[:, 7].astype(np.int16))
        out = tracker.update(d, (h, w))
        collect(tracks, f / fps, out)
    vid = npz_path.name.split("__")[0]
    H = np.load(npz_path.parents[1] / "align" / f"{vid}.npz")["H"] @ np.diag([REF_SIZE[0] / w, REF_SIZE[1] / h, 1.0])
    return build(tracks, H), {"H": H, "fps": fps, "stride": stride, "width": w, "height": h, "n_frames": int(z["n_frames"])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+")
    ap.add_argument("--out", default="cache/tracks")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for c in args.caches:
        t0 = time.perf_counter()
        trajs, meta = track_cache(Path(c))
        vid = Path(c).name.split("__")[0]
        with open(Path(args.out) / f"{vid}.pkl", "wb") as fh:
            pickle.dump({"meta": meta, "trajectories": trajs}, fh)
        groups = {}
        for tr in trajs:
            groups[tr.group] = groups.get(tr.group, 0) + 1
        print(f"{vid}: {len(trajs)} trajectories {groups} in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
