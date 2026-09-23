"""Collect verified labels from the labelling workflow runs into labels/dev_labels.json.

    python tools/build_dev_labels.py <workflow journal.jsonl> [...] --out labels/dev_labels.json

Keeps claims a verifier confirmed or relabelled (with the verifier's class and
boundaries), merges same-class segments that overlap (the task's convention
for simultaneous events), and writes the organisers' ground-truth format.
Unverified or rejected claims go to labels/dev_labels_rejected.json for review.

    --adjudication labels/adjudication.json

folds in the adjudication pass (tools/workflows/adjudicate.js, one agent per
disagreement between the model and these labels): an event the labels missed is
added, a label the adjudicator found wrong is removed, and a boundary-only
disagreement takes the adjudicator's times when the labels it touches form one run.
Only disagreements get a second look, so this favours the model somewhat; the
report says so.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parivision.segments import union  # noqa: E402

DURATIONS = {"C3896": 340.34, "C3897": 317.82, "C3902": 317.82, "C3905": 127.63}
FPS = 29.97


def _overlaps(a, b) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def adjudicate(kept: dict[str, list], items: list[dict]) -> None:
    changes = {"added": 0, "removed": 0, "retimed": 0}
    for it in items:
        v = it.get("verdict")
        if not v:
            continue
        claims = kept.setdefault(it["clip"], [])
        seg = (it["start"], it["end"])
        cat = v["category"]
        if it["kind"] == "fp" and cat == "label_missed_it" and v["real_event"] and v["end"] > v["start"]:
            claims.append([round(v["start"], 2), round(v["end"], 2), v["label"]])
            changes["added"] += 1
        elif it["kind"] == "fn" and cat == "label_wrong" and not v["real_event"]:
            before = len(claims)
            claims[:] = [c for c in claims if not (c[2] == it["label"] and _overlaps(c, seg))]
            changes["removed"] += before - len(claims)
        elif cat == "boundary_only" and v["real_event"] and v["end"] > v["start"]:
            # claims from both specialists often describe the same event; retime only if they form one run
            hit = [c for c in claims if c[2] == v["label"] and _overlaps(c, (v["start"], v["end"]))]
            if hit and len(union([(c[0], c[1]) for c in hit])) == 1:
                claims[:] = [c for c in claims if not any(c is h for h in hit)]
                claims.append([round(v["start"], 2), round(v["end"], 2), v["label"]])
                changes["retimed"] += 1
    print("adjudication:", changes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("journals", nargs="+")
    ap.add_argument("--out", default="labels/dev_labels.json")
    ap.add_argument("--adjudication", help="result of the adjudication workflow (list of items with verdicts)")
    args = ap.parse_args()

    kept: dict[str, list] = {}
    dropped: dict[str, list] = {}
    for jpath in args.journals:
        started = {}
        for line in open(jpath):
            d = json.loads(line)
            if d["type"] == "started":
                started[d["key"]] = d.get("label", "")
            if d["type"] != "result" or not isinstance(d.get("result"), dict):
                continue
            r = d["result"]
            label = started.get(d["key"], "")
            m = re.match(r"verify:(C\d+):", label)
            if "verdict" in r and m:
                clip = m.group(1)
                if r["verdict"] in ("confirmed", "relabelled") and r["end"] > r["start"]:
                    kept.setdefault(clip, []).append([round(r["start"], 2), round(r["end"], 2), r["label"]])
                else:
                    dropped.setdefault(clip, []).append({**r})
    if args.adjudication:
        adjudicate(kept, json.loads(Path(args.adjudication).read_text()))
    gt = {}
    for clip, dur in DURATIONS.items():
        per_class: dict[str, list] = {}
        for s, e, lab in kept.get(clip, []):
            per_class.setdefault(lab, []).append((max(0.0, s), min(dur, e)))
        events = sorted([round(s, 2), round(e, 2), lab] for lab, ivs in per_class.items() for s, e in union(ivs))
        gt[f"{clip}.MP4"] = {"duration": dur, "fps": FPS, "events": events}
        print(clip, len(events), {lab: sum(1 for x in events if x[2] == lab) for lab in sorted(per_class)})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(gt, indent=1))
    Path(args.out).with_name("dev_labels_rejected.json").write_text(json.dumps(dropped, indent=1))


if __name__ == "__main__":
    main()
