"""Render an annotated MP4 for a sample clip from cached tracks (dev loop; the website uses tools/make_site_data.py).

    python tools/render_clip.py C3905 --seconds 30 --out out/render/C3905.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from common import SAMPLES  # noqa: E402
from run_rules import load_context  # noqa: E402

from parivision.events import detect_from_context  # noqa: E402
from parivision.render import render  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ctx, H = load_context(args.clip)
    events, evidence = detect_from_context(ctx, H)
    out = args.out or str(ROOT / "out/render" / f"{args.clip}.mp4")
    render(str(SAMPLES / f"{args.clip}.MP4"), out, ctx.trajectories, events, evidence,
           ctx.signal_t, ctx.signal_phase, H, work_width=1920, max_seconds=args.seconds)
    print(out)


if __name__ == "__main__":
    main()
