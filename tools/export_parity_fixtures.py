"""Fixtures that pin the in-browser pipeline (site/src/pipeline/) to the Python one.

    python tools/export_parity_fixtures.py [--clip C3905] [--tag yolo26m_1280_1920_10fps] [--every 2]

Writes site/tests/fixtures/<clip>/ (gitignored; rebuilt by this script):

  detections.json   the cached detector output at the demo's frame rate: frames of [x1,y1,x2,y2,conf,cls]
                    in 1920-wide working pixels
  tracks.json       what the Python MultiTracker returns for those frames: rows [x1,y1,x2,y2,tid,score,cls]
  trajectories.json trajectories.build() of those tracks (reference view)
  signal.json       the lamp scores, raw and filled phases of the clip (cache/signal)
  alignment.json    the clip's homography (working pixels -> reference) and the grey 480x270 first frame
                    it was computed from, as a PNG next to it (frame0.png)
  lamps.json        lamp patch boxes and scores for a few full-size frames saved as lamp_<t>.png
  rules.json        Python evidence and final events for trajectories.json + signal.json
  risk.json         Part B's risk curve replayed over detections.json (Anticipator.observe)

Each stage's input is the previous stage's Python output, so a JS module can be checked on its
own: tracker on detections, build on tracks, rules on trajectories, and so on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parivision import events as E  # noqa: E402
from parivision import scene as S  # noqa: E402
from parivision.detector import Detections  # noqa: E402
from parivision.registration import Alignment  # noqa: E402
from parivision.risk import Anticipator  # noqa: E402
from parivision.rules import Context  # noqa: E402
from parivision.signal import lamp_patches, lamp_scores  # noqa: E402
from parivision.tracking import MultiTracker, collect  # noqa: E402
from parivision.trajectories import build  # noqa: E402
from parivision.video import sample_frames  # noqa: E402


def r(x, n=3):
    return np.round(np.asarray(x, np.float64), n).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="C3905")
    ap.add_argument("--tag", default="yolo26m_1280_1920_10fps")
    ap.add_argument("--every", type=int, default=2, help="keep every n-th cached frame (2: 10 fps -> 5 fps)")
    ap.add_argument("--seconds", type=float, default=0, help="only the first N seconds (0: whole clip)")
    args = ap.parse_args()
    out = ROOT / "site/tests/fixtures" / args.clip
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(ROOT / "cache/det" / f"{args.clip}__{args.tag}.npz")
    det, fps, stride = z["det"], float(z["fps"]), int(z["stride"]) * args.every
    w, h = int(z["width"]), int(z["height"])
    n_frames = int(z["n_frames"])
    if args.seconds:
        n_frames = min(n_frames, int(args.seconds * fps))
    by_frame = {int(f): det[det[:, 0] == f] for f in np.unique(det[:, 0])}

    frames, track_rows, tracks = [], [], {}
    tracker = MultiTracker(fps=fps / stride)
    H = np.load(ROOT / "cache/align" / f"{args.clip}.npz")["H"] @ np.diag([S.REF_SIZE[0] / w, S.REF_SIZE[1] / h, 1.0])
    risk_model = Anticipator({"fps": fps / stride, "width": w, "height": h})
    risk_model.alignment = Alignment(H, 1, True)
    risk = []
    for f in range(0, n_frames, stride):
        rows = by_frame.get(f)
        d = Detections.empty() if rows is None else Detections(
            rows[:, 2:6].astype(np.float32), rows[:, 6].astype(np.float32), rows[:, 7].astype(np.int16))
        t = f / fps
        frames.append({"t": round(t, 4), "boxes": [r(list(b) + [c, k]) for b, c, k in zip(d.xyxy, d.conf, d.cls)]})
        out_rows = tracker.update(d, (h, w))
        track_rows.append({"t": round(t, 4), "rows": [r(x) for x in out_rows]})
        collect(tracks, t, out_rows)
        risk.append([round(t, 3), round(risk_model.observe(d, (h, w), t), 5)])
    duration = n_frames / fps
    (out / "detections.json").write_text(json.dumps({"clip": args.clip, "fps": fps / stride, "width": w, "height": h,
                                                     "duration": duration, "frames": frames}))
    (out / "tracks.json").write_text(json.dumps({"fps": fps / stride, "frames": track_rows}))
    (out / "risk.json").write_text(json.dumps({"risk": risk}))

    trajs = build(tracks, H)
    (out / "trajectories.json").write_text(json.dumps([{
        "tid": tr.tid, "group": tr.group, "cls": tr.cls, "t": r(tr.t, 4), "box": r(tr.box, 2), "foot": r(tr.foot, 3),
        "vel": r(tr.vel, 3), "height": r(tr.height, 3), "conf": r(tr.conf, 4)} for tr in trajs]))

    sig = json.loads((ROOT / "cache/signal" / f"{args.clip}.json").read_text())
    (out / "signal.json").write_text(json.dumps({k: sig[k] for k in ("times", "raw", "phases", "scores")}))
    ctx = Context(trajs, np.array(sig["times"]), np.array(sig["phases"]), duration)
    events, evidence = E.detect_from_context(ctx, H)
    (out / "rules.json").write_text(json.dumps({
        "dt": ctx.dt, "people": sorted(tr.tid for tr in ctx.people), "events": events,
        "evidence": [{"label": e.label, "start": round(e.start, 4), "end": round(e.end, 4), "actors": e.actors,
                      "note": e.note} for e in evidence]}, indent=1))

    # alignment input and the lamp reader on real pixels
    video = ROOT / "kit/samples" / f"{args.clip}.MP4"
    lamp_t = [1.0, 40.0, 76.0, 100.0]
    lamps, first = [], None
    boxes = lamp_patches(np.linalg.inv(H))
    for _, t, frame in sample_frames(video, 1.0, w):
        if first is None:
            first = frame
            cv2.imwrite(str(out / "frame0.png"), cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (480, 270),
                                                            interpolation=cv2.INTER_AREA))
        if any(abs(t - x) < 0.5 for x in lamp_t):
            name = f"lamp_{int(round(t))}.png"
            x0 = min(b[0] for b in boxes) - 20
            y0 = min(b[1] for b in boxes) - 20
            crop = frame[y0:max(b[3] for b in boxes) + 20, x0:max(b[2] for b in boxes) + 20]
            cv2.imwrite(str(out / name), crop)
            lamps.append({"file": name, "t": round(t, 3), "origin": [x0, y0], "scores": r(lamp_scores(frame, boxes), 3)})
        if t > max(lamp_t) + 1:
            break
    (out / "alignment.json").write_text(json.dumps({"H": r(H, 8), "frame": "frame0.png", "frame_size": [480, 270],
                                                    "work_size": [w, h]}))
    (out / "lamps.json").write_text(json.dumps({"boxes": [list(map(int, b)) for b in boxes], "frames": lamps}))
    print(f"{args.clip}: {len(frames)} frames, {len(trajs)} trajectories, {len(events)} events -> {out}")


if __name__ == "__main__":
    main()
