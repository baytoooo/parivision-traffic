"""Contact sheets from a proxy video, for eyeballing events and labelling.

    python tools/sheet.py cache/proxy/C3905.mp4 --start 30 --end 60 --step 2 --cols 4 --out out/sheets/x.jpg
    python tools/sheet.py cache/proxy/C3905.mp4 --start 30 --end 36 --step 0.5 --crop 0,200,960,540

--crop x0,y0,x1,y1 is in proxy pixels (960x540) and lets you zoom on one area.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, required=True)
    ap.add_argument("--end", type=float, required=True)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--tile", type=int, default=480, help="tile width in pixels")
    ap.add_argument("--crop", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    tiles = []
    t = args.start
    while t <= args.end + 1e-6:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
        ok, frame = cap.read()
        if not ok:
            break
        if args.crop:
            x0, y0, x1, y1 = map(int, args.crop.split(","))
            frame = frame[y0:y1, x0:x1]
        scale = args.tile / frame.shape[1]
        tile = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        label = f"{t:.1f}s"
        cv2.rectangle(tile, (0, 0), (8 + 11 * len(label), 22), (0, 0, 0), -1)
        cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
        t += args.step
    if not tiles:
        raise SystemExit("no frames in range")
    h, w = tiles[0].shape[:2]
    while len(tiles) % args.cols:
        tiles.append(np.zeros((h, w, 3), np.uint8))
    rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
    sheet = np.vstack(rows)
    out = Path(args.out or f"out/sheets/{Path(args.video).stem}_{args.start:.0f}_{args.end:.0f}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)


if __name__ == "__main__":
    main()
