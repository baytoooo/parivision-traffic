"""Run the detector over sample videos once and store raw detections.

Rule development then iterates on the cache (tracking takes seconds) instead
of decoding 4K video again.

    python tools/cache_detections.py samples/*.MP4 --fps 10 --width 1920
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parivision.detector import Detector  # noqa: E402
from parivision.video import probe, sample_frames  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="cache/det")
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--weights", default="yolo26m.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    det = Detector(args.weights, imgsz=args.imgsz, conf=args.conf)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(args.weights).stem}_{args.imgsz}_{args.width}_{args.fps:g}fps"

    for v in args.videos:
        info = probe(v)
        target = out_dir / f"{Path(v).stem}__{tag}.npz"
        if target.exists():
            print("skip", target)
            continue
        t0 = time.perf_counter()
        frames, idxs, times, rows = [], [], [], []

        def flush() -> None:
            for (fi, ts), d in zip(zip(idxs, times), det(frames)):
                for box, c, k in zip(d.xyxy, d.conf, d.cls):
                    rows.append((fi, ts, *box, c, k))
            frames.clear(); idxs.clear(); times.clear()

        n = 0
        for fi, ts, img in sample_frames(v, args.fps, args.width):
            frames.append(img); idxs.append(fi); times.append(ts)
            n += 1
            if len(frames) == args.batch:
                flush()
            if n % 500 == 0:
                el = time.perf_counter() - t0
                print(f"  {Path(v).name} t={ts:.0f}s/{info.duration:.0f}s  {el:.0f}s elapsed", flush=True)
        flush()
        arr = np.array(rows, dtype=np.float64).reshape(-1, 8)  # frame, t, x1, y1, x2, y2, conf, cls
        np.savez_compressed(
            target, det=arr, fps=info.fps, width=args.width,
            height=int(round(info.height * args.width / info.width)), n_frames=info.n_frames,
            sampled=n, stride=max(1, int(round(info.fps / args.fps))),
        )
        dt = time.perf_counter() - t0
        print(f"{Path(v).name}: {n} frames, {len(arr)} boxes, {dt:.0f}s ({info.duration / dt:.2f}x realtime) -> {target}")


if __name__ == "__main__":
    main()
