"""Show which actors produced an event: prints the evidence and renders proxy frames with their tracks.

    python tools/inspect_evidence.py C3896 jaywalking --t 100
    python tools/inspect_evidence.py C3896 stopped_vehicle --all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from run_rules import load_context  # noqa: E402

from parivision.events import detect_from_context  # noqa: E402
from parivision.registration import warp_points  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("label")
    ap.add_argument("--t", type=float, default=None, help="only evidence covering this time")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max", type=int, default=12)
    ap.add_argument("--render", type=int, default=3, help="render this many evidence items")
    args = ap.parse_args()

    ctx, H = load_context(args.video)
    _, evidence = detect_from_context(ctx, H)
    items = [e for e in evidence if e.label == args.label and (args.t is None or e.start <= args.t <= e.end)]
    items.sort(key=lambda e: e.start)
    for e in items[: args.max if not args.all else None]:
        desc = []
        for tid in e.actors:
            tr = ctx.by_id[tid]
            i0 = int(np.argmin(np.abs(tr.t - e.start)))
            desc.append(f"{tid}(cls{tr.cls} @ {tr.foot[i0].round().tolist()} v={tr.speed[i0]:.0f})")
        print(f"{e.start:7.1f} {e.end:7.1f} {e.note:12s} " + " ".join(desc))

    cap = cv2.VideoCapture(str(ROOT / "cache/proxy" / f"{args.video}.mp4"))
    out_dir = ROOT / "out/inspect"
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, e in enumerate(items[: args.render]):
        tiles = []
        for t in np.linspace(e.start, e.end, 4):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * 10)))
            ok, frame = cap.read()
            if not ok:
                continue
            for tid in e.actors:
                tr = ctx.by_id[tid]
                pts = (warp_points(tr.foot, np.linalg.inv(H)) / 2).astype(np.int32)
                cv2.polylines(frame, [pts], False, (0, 255, 255), 1)
                i = tr.at(t)
                if i is not None:
                    x1, y1, x2, y2 = (tr.box[i] / 2).astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{t:.1f}s", (8, 22), 0, 0.7, (255, 255, 255), 2)
            tiles.append(frame)
        if tiles:
            while len(tiles) < 4:
                tiles.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4])])
            path = out_dir / f"{args.video}_{args.label}_{k}.jpg"
            cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
            print("rendered", path)


if __name__ == "__main__":
    main()
