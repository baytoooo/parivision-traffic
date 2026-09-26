"""Compare two prediction files video by video: events per class and how many of them match.

    python tools/compare_predictions.py predictions_samples.json predictions_samples_rerun.json

For each video it prints the number of events per class in both files and, at each tIoU threshold
evaluate.py scores at (0.3, 0.5, 0.7), how many events of that class pair up across the two files.
The pairing is evaluate.py's own match_segments (greedy, one to one, by descending tIoU). The match
rate is the share of all events in both files that found a partner: 1.0 when every event in one
file has a partner of the same class in the other. A video that is in only one file counts toward
the overall rate with none of its events matched. It also prints the largest difference between
the two risk curves at the times both files have a sample for.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluate import TIOU_THRESHOLDS, match_segments, segs  # noqa: E402


def matched(a: list, b: list, label: str, thr: float) -> int:
    tp, _, _ = match_segments(segs(a, label), segs(b, label), thr)
    return tp


def risk_diff(a: list, b: list) -> tuple[int, float | None]:
    """(samples at a shared time, largest |score difference| there)."""
    ra = {round(float(t), 3): float(s) for t, s in a}
    shared = [(ra[round(float(t), 3)], float(s)) for t, s in b if round(float(t), 3) in ra]
    return len(shared), max((abs(x - y) for x, y in shared), default=None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="first prediction file, e.g. the committed predictions_samples.json")
    ap.add_argument("b", help="second prediction file, e.g. your rerun")
    args = ap.parse_args()
    va = json.loads(Path(args.a).read_text())["videos"]
    vb = json.loads(Path(args.b).read_text())["videos"]
    total = {thr: 0 for thr in TIOU_THRESHOLDS}
    n_all = 0
    width = max([len("all classes")] + [len(e[2]) for v in (*va.values(), *vb.values()) for e in v.get("events", [])])
    for video in sorted(set(va) | set(vb)):
        if video not in va or video not in vb:
            only, path = (va, args.a) if video in va else (vb, args.b)
            n = len(only[video].get("events", []))
            print(f"{video}: only in {path}, {n} events without a partner\n")
            n_all += n
            continue
        ea, eb = va[video].get("events", []), vb[video].get("events", [])
        print(video)
        print(f"  {'class':{width}s}  {'a':>4s} {'b':>4s}  matched at tIoU " + " / ".join(f"{t:g}" for t in TIOU_THRESHOLDS))
        tp = {thr: 0 for thr in TIOU_THRESHOLDS}
        for label in sorted({e[2] for e in ea} | {e[2] for e in eb}):
            m = [matched(ea, eb, label, thr) for thr in TIOU_THRESHOLDS]
            for thr, k in zip(TIOU_THRESHOLDS, m):
                tp[thr] += k
            na, nb = len(segs(ea, label)), len(segs(eb, label))
            print(f"  {label:{width}s}  {na:4d} {nb:4d}  " + " / ".join(str(k) for k in m))
        n = len(ea) + len(eb)
        rates = " / ".join(f"{2 * tp[thr] / n:.3f}" if n else "-" for thr in TIOU_THRESHOLDS)
        print(f"  {'all classes':{width}s}  {len(ea):4d} {len(eb):4d}  match rate {rates}")
        k, d = risk_diff(va[video].get("risk", []), vb[video].get("risk", []))
        print(f"  risk: {len(va[video].get('risk', []))} and {len(vb[video].get('risk', []))} samples, "
              + (f"largest difference {d:.4f} over {k} shared times" if d is not None else "no shared times") + "\n")
        for thr in TIOU_THRESHOLDS:
            total[thr] += tp[thr]
        n_all += n
    if n_all:
        print("all videos: match rate " + " / ".join(f"{2 * total[thr] / n_all:.3f}" for thr in TIOU_THRESHOLDS)
              + " at tIoU " + " / ".join(f"{t:g}" for t in TIOU_THRESHOLDS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
