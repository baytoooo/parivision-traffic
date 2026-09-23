"""Print basic metadata for videos and dump evenly spaced frames for a first look.

    python tools/probe.py samples/*.MP4 --out out/probe --every 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="out/probe")
    ap.add_argument("--every", type=float, default=30.0, help="seconds between dumped frames")
    ap.add_argument("--width", type=int, default=1280, help="resize dumped frames to this width")
    args = ap.parse_args()

    for v in args.videos:
        path = Path(v)
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))
        print(f"{path.name}: {w}x{h} @ {fps:.3f} fps, {n} frames, {n / fps:.1f}s, codec {codec}, "
              f"{path.stat().st_size / 1e9:.2f} GB")
        out = Path(args.out) / path.stem
        out.mkdir(parents=True, exist_ok=True)
        step = max(1, int(round(args.every * fps)))
        for idx in range(0, n, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                break
            scale = args.width / frame.shape[1]
            small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out / f"t{idx / fps:07.1f}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
        cap.release()


if __name__ == "__main__":
    main()
