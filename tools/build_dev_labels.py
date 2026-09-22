"""Collect verified labels from the labelling workflow runs into labels/dev_labels.json.

    python tools/build_dev_labels.py <workflow journal.jsonl> [...] --out labels/dev_labels.json

Keeps claims a verifier confirmed or relabelled (with the verifier's class and
boundaries), merges same-class segments that overlap (the task's convention
for simultaneous events), and writes the organisers' ground-truth format.
Unverified or rejected claims go to labels/dev_labels_rejected.json for review.
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("journals", nargs="+")
    ap.add_argument("--out", default="labels/dev_labels.json")
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
