"""Run the event rules on cached trajectories (fast loop for tuning against dev labels).

    python tools/run_rules.py C3896 C3905 --out out/pred_dev.json
    python evaluate.py --pred out/pred_dev.json --gt labels/dev_labels.json --per-video
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parivision.events import detect_from_context  # noqa: E402
from parivision.rules import Context  # noqa: E402


def load_context(vid: str) -> tuple[Context, np.ndarray]:
    d = pickle.load(open(ROOT / "cache/tracks" / f"{vid}.pkl", "rb"))
    sig = json.loads((ROOT / "cache/signal" / f"{vid}.json").read_text())
    meta = d["meta"]
    ctx = Context(d["trajectories"], np.array(sig["times"]), np.array(sig["phases"]),
                  meta["n_frames"] / meta["fps"])
    return ctx, meta["H"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out", default="out/pred_dev.json")
    args = ap.parse_args()
    pred = {"team": "PariVision", "videos": {}}
    for vid in args.videos:
        ctx, H = load_context(vid)
        events, evidence = detect_from_context(ctx, H)
        pred["videos"][f"{vid}.MP4"] = {"events": events, "risk": []}
        print(vid, dict(Counter(e[2] for e in events)))
        for e in events:
            print(f"   {e[0]:7.1f} {e[1]:7.1f}  {e[2]}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(pred, indent=1))


if __name__ == "__main__":
    main()
