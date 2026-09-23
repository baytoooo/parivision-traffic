"""Grid search one class's rule parameters against the dev labels (in-process, contexts loaded once).

    python tools/tune.py failure_to_yield --gt labels/dev_labels.json --clips C3896 C3905

Prints per-class mean F1 (tIoU 0.3/0.5/0.7) for every combination, best first,
split by clip so a change that only helps one clip stands out.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from evaluate import evaluate  # noqa: E402
from run_rules import load_context  # noqa: E402

from parivision import events as E  # noqa: E402
from parivision import rules as R  # noqa: E402
from parivision.segments import finalize  # noqa: E402

GRIDS = {
    "failure_to_yield": {
        "gap": [0.0, 0.3, 1.0],
        "fty_near_px": [160, 220],
        "fty_walk": [0.0, 0.25],
        "fty_kerb": [0.4],
        "fty_pad_start": [0.0, 0.3],
        "fty_moto_speed": [float("inf"), 60.0, 120.0],
    },
    "jaywalking": {
        "gap": [0.5, 1.5, 3.0],
        "jay_margin_cw": [0.2, 0.35, 0.5],
        "jay_margin_kerb": [0.15, 0.25, 0.4],
        "jay_min_dur": [0.5, 1.0],
    },
}


def run_class(ctx, H, label):
    if label == "failure_to_yield":
        return R.failure_to_yield(ctx, H)
    if label == "jaywalking":
        return R.jaywalking(ctx)
    raise KeyError(label)


def score(gt, pred, label):
    rep = evaluate(gt, pred)
    return rep["part_a"]["per_class"].get(label, {}).get("f1_mean", 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("--gt", default="labels/dev_labels.json")
    ap.add_argument("--clips", nargs="+", default=["C3896", "C3897", "C3902", "C3905"])
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()
    gt_all = json.loads((ROOT / args.gt).read_text())
    ctxs = {c: load_context(c) for c in args.clips}
    grid = GRIDS[args.label]
    keys = list(grid)
    base = dict(R.PARAMS)
    rows = []
    for combo in itertools.product(*grid.values()):
        opt = dict(zip(keys, combo))
        R.PARAMS.clear()
        R.PARAMS.update(base)
        R.PARAMS.update({k: v for k, v in opt.items() if k != "gap"})
        gap = {**E.GAP, args.label: opt["gap"]}
        per_clip = {}
        pred_all = {"team": "t", "videos": {}}
        for c, (ctx, H) in ctxs.items():
            ivs = [(ev.start, ev.end) for ev in run_class(ctx, H, args.label)]
            events = finalize({args.label: ivs}, ctx.duration, gap, E.MIN_LEN)
            key = f"{c}.MP4"
            pred_all["videos"][key] = {"events": events, "risk": []}
            g = {key: {**gt_all[key], "events": [x for x in gt_all[key]["events"] if x[2] == args.label]}}
            per_clip[c] = score(g, {"team": "t", "videos": {key: pred_all["videos"][key]}}, args.label)
        g_all = {f"{c}.MP4": {**gt_all[f"{c}.MP4"], "events": [x for x in gt_all[f"{c}.MP4"]["events"]
                                                             if x[2] == args.label]} for c in ctxs}
        rows.append((score(g_all, pred_all, args.label), per_clip, opt))
    R.PARAMS.clear()
    R.PARAMS.update(base)
    rows.sort(key=lambda r: -r[0])
    cur = {k: (E.GAP[args.label] if k == "gap" else base[k]) for k in keys}
    for f, per_clip, opt in rows[: args.top]:
        mark = " <- current" if opt == cur else ""
        print(f"{f:.3f}  " + " ".join(f"{c}:{v:.2f}" for c, v in per_clip.items()) + f"  {opt}{mark}")
    for f, per_clip, opt in rows:
        if opt == cur:
            print("current:", f"{f:.3f}", " ".join(f"{c}:{v:.2f}" for c, v in per_clip.items()))


if __name__ == "__main__":
    main()
