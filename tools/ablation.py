"""Ablations on the dev set: detector, frame rate, and the design choices we made.

    python tools/ablation.py --gt labels/dev_labels.json --out out/ablations.json

Every variant re-tracks cached detections and re-runs the rules, then scores with
the organisers' evaluate.py, on every clip that has the detections it needs
(tools/cache_detections.py --weights ... --imgsz ...). A variant scored on fewer
clips also gets the submitted configuration scored on the same clips, so each
row has a like-for-like baseline.

Two numbers per row: the official Score A, and the mean F1 over the classes we
emit. The first also averages in zeros for classes our labels have and we never
predict (U-turns, illegal turns), which are the same for every variant.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evaluate import evaluate  # noqa: E402

from parivision import events as E  # noqa: E402
from parivision import scene as S  # noqa: E402
from parivision.detector import Detections  # noqa: E402
from parivision.rules import Context  # noqa: E402
from parivision.tracking import MultiTracker, collect  # noqa: E402
from parivision.trajectories import build  # noqa: E402

VARIANTS = [
    # name, detector cache tag, keep every n-th cached frame, options, note for the site
    ("YOLO26m @1280, 10 fps", "yolo26m_1280_1920_10fps", 1, {},
     "Submitted configuration."),
    ("YOLO26m @1280, 5 fps", "yolo26m_1280_1920_10fps", 2, {},
     "Every 6th frame: half the detector work."),
    ("YOLO26m @1280, 3.3 fps", "yolo26m_1280_1920_10fps", 3, {},
     "Every 9th frame: a third of the detector work."),
    ("YOLO26s @1280, 10 fps", "yolo26s_1280_1920_10fps", 1, {},
     "Smaller detector (the Part B one), same input size."),
    ("YOLO26m @960, 10 fps", "yolo26m_960_1920_10fps", 1, {},
     "Same detector, 960 px input: people near the top of the frame shrink to about 17 px."),
    ("No registration", "yolo26m_1280_1920_10fps", 1, {"no_registration": True},
     "Scene polygons used as drawn, without mapping each clip onto the reference view."),
    ("Hand-drawn road mask", "yolo26m_1280_1920_10fps", 1, {"hand_road": True},
     "Jaywalking checked against the traced carriageway instead of the drivable area learned from vehicle tracks."),
]
# detector compute relative to the submitted run: input pixels x frames
REL_COST = {"yolo26m_1280": 1.0, "yolo26s_1280": 0.46, "yolo26m_960": 0.56}


def trajectories(clip: str, tag: str, every: int, no_registration: bool):
    z = np.load(ROOT / "cache/det" / f"{clip}__{tag}.npz")
    det, fps, stride = z["det"], float(z["fps"]), int(z["stride"]) * every
    w, h = int(z["width"]), int(z["height"])
    tracker = MultiTracker(fps=fps / stride)
    tracks: dict = {}
    by_frame = {int(f): det[det[:, 0] == f] for f in np.unique(det[:, 0])}
    for f in range(0, int(z["n_frames"]), stride):
        rows = by_frame.get(f)
        d = Detections.empty() if rows is None else Detections(
            rows[:, 2:6].astype(np.float32), rows[:, 6].astype(np.float32), rows[:, 7].astype(np.int16))
        collect(tracks, f / fps, tracker.update(d, (h, w)))
    scale = np.diag([S.REF_SIZE[0] / w, S.REF_SIZE[1] / h, 1.0])
    H = scale if no_registration else np.load(ROOT / "cache/align" / f"{clip}.npz")["H"] @ scale
    return build(tracks, H), H, int(z["n_frames"]) / fps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", default=["C3896", "C3897", "C3902", "C3905"])
    ap.add_argument("--gt", default="labels/dev_labels.json")
    ap.add_argument("--out", default="out/ablations.json")
    args = ap.parse_args()
    gt = json.loads((ROOT / args.gt).read_text())
    rows = []
    base_tag = VARIANTS[0][1]

    def score(tag, every, opt, clips):
        pred = {"team": "ablation", "videos": {}}
        for clip in clips:
            trajs, H, duration = trajectories(clip, tag, every, opt.get("no_registration", False))
            sig = json.loads((ROOT / "cache/signal" / f"{clip}.json").read_text())
            masks = S.masks()
            if opt.get("hand_road"):
                masks["walk_check"] = masks["road"]
            ctx = Context(trajs, np.array(sig["times"]), np.array(sig["phases"]), duration, masks=masks)
            events, _ = E.detect_from_context(ctx, H)
            pred["videos"][f"{clip}.MP4"] = {"events": events, "risk": []}
        rep = evaluate({f"{c}.MP4": gt[f"{c}.MP4"] for c in clips}, pred)["part_a"]
        per = {c: round(v["f1_mean"], 3) for c, v in rep["per_class"].items()}
        emitted = [v for c, v in per.items() if c in E.ENABLED]
        return round(rep["score_a"], 4), round(float(np.mean(emitted)), 4) if emitted else None, per

    for name, tag, every, opt, note in VARIANTS:
        clips = [c for c in args.clips if (ROOT / "cache/det" / f"{c}__{tag}.npz").exists()]
        if not clips:
            print("skip (no cache):", name)
            continue
        if tag != base_tag and len(clips) < len(args.clips) and not any(r["clips"] == clips for r in rows[1:]):
            a, m, per = score(base_tag, 1, {}, clips)
            rows.append({"name": f"{VARIANTS[0][0]}, same clips", "clips": clips, "score_a": a, "emitted_mean": m,
                         "cost_x": 1.0, "note": "Baseline for the row below.", "per_class": per})
        a, m, per = score(tag, every, opt, clips)
        det_key = "_".join(tag.split("_")[:2])
        rows.append({"name": name, "clips": clips, "score_a": a, "emitted_mean": m,
                     "cost_x": round(REL_COST.get(det_key, 1.0) / every, 2), "note": note, "per_class": per})
        print(f"{name:28s} {len(clips)} clips  Score A {a:.3f}  emitted-class mean {m:.3f}  cost x{rows[-1]['cost_x']}")
    Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
