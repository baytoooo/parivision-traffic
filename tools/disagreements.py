"""List every disagreement between predictions and dev labels, as input for the adjudication workflow.

    python tools/disagreements.py --gt labels/dev_labels.json --out out/adjudicate_items.json

Runs the rules again on cache/tracks and cache/signal (tools/tracks_from_cache.py,
tools/signal_timeline.py), so it reads no predictions file.

A prediction with no label overlapping it at tIoU >= 0.3 is a false-positive candidate; a
label with no prediction at tIoU >= 0.3 is a miss candidate. Actor ids and positions from
the rule evidence are attached so the adjudicator knows whom to look at.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from run_rules import load_context  # noqa: E402

from parivision.events import detect_from_context  # noqa: E402


def tiou(a, b) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def signal_text(ctx, s: float, e: float) -> str:
    segs = []
    for t, p in zip(ctx.signal_t, ctx.signal_phase):
        if s - 10 <= t <= e + 10:
            if segs and segs[-1][2] == p:
                segs[-1][1] = t
            else:
                segs.append([t, t, p])
    return "; ".join(f"{a:.1f}-{b:.1f} s {p}" for a, b, p in segs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default="labels/dev_labels.json")
    ap.add_argument("--out", default="out/adjudicate_items.json")
    ap.add_argument("--thr", type=float, default=0.3)
    args = ap.parse_args()
    gt = json.loads((ROOT / args.gt).read_text())
    items = []
    for key, g in gt.items():
        clip = Path(key).stem
        ctx, H = load_context(clip)
        events, evidence = detect_from_context(ctx, H)
        labels = g["events"]
        for s, e, lab in events:
            if max((tiou((s, e), (a, b)) for a, b, lb in labels if lb == lab), default=0.0) >= args.thr:
                continue
            actors = []
            for ev in evidence:
                if ev.label == lab and ev.start < e and ev.end > s:
                    for tid in ev.actors[:4]:
                        tr = ctx.by_id.get(tid)
                        if tr is not None:
                            i = tr.at((max(s, ev.start) + min(e, ev.end)) / 2) or 0
                            x, y = (tr.box[i][:2] + tr.box[i][2:]) / 2 / 2  # proxy pixels (approximate)
                            actors.append(f"{tr.group} box centre near proxy ({x:.0f},{y:.0f}) at {tr.t[i]:.1f} s")
            items.append({"clip": clip, "kind": "fp", "label": lab, "start": s, "end": e,
                          "actors": "; ".join(actors[:6]) or "n/a", "signal": signal_text(ctx, s, e)})
        for s, e, lab in labels:
            if max((tiou((s, e), (a, b)) for a, b, lb in events if lb == lab), default=0.0) >= args.thr:
                continue
            items.append({"clip": clip, "kind": "fn", "label": lab, "start": s, "end": e, "actors": "",
                          "signal": signal_text(ctx, s, e)})
    for k, it in enumerate(items):
        it["id"] = f"{it['clip']}_{it['kind']}_{k}"
    Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(json.dumps(items, indent=1))
    from collections import Counter

    print(len(items), Counter((i["kind"], i["label"]) for i in items))


if __name__ == "__main__":
    main()
