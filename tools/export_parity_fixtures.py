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
  rules_perturbed.json  the same for three perturbed inputs (phases rolled, slow traffic on green,
                    every 10th trajectory played backwards), so red_light, congestion and wrong_way fire
  risk.json         Part B's risk curve replayed over detections.json (Anticipator.observe)
  scene_samples.json  mask bits, zone bits and distances at 3000 random reference points
  det_frame.rgb     one frame at 960x540, raw RGB, and det_expected.json: what Ultralytics gets from
                    site/public/pipeline/model.onnx on it (boxes in 960x540 pixels)

Images are also written as raw bytes (.gray, .rgb) so the Node tests need no image decoder;
site/tests/fixtures/refs/ holds the two references as raw grey 480x270.

Trajectories are written at full precision (event edges are rounded to 0.01 s, and a value cut to a
few decimals can land on a rounding tie, or move an interpolated crossing time, where the real one does not). Each stage's input is the previous stage's Python output, so a JS module can be checked on its
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
sys.path.insert(0, str(ROOT / "tools"))

from common import SAMPLES  # noqa: E402
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
        frames.append({"t": t, "boxes": [[*map(float, b), float(c), int(k)] for b, c, k in zip(d.xyxy, d.conf, d.cls)]})
        out_rows = tracker.update(d, (h, w))
        track_rows.append({"t": t, "rows": [r(x) for x in out_rows]})
        collect(tracks, t, out_rows)
        risk.append([round(t, 3), round(risk_model.observe(d, (h, w), t), 5)])
    duration = n_frames / fps
    (out / "detections.json").write_text(json.dumps({"clip": args.clip, "fps": fps / stride, "width": w, "height": h,
                                                     "duration": duration, "frames": frames}))
    (out / "tracks.json").write_text(json.dumps({"fps": fps / stride, "frames": track_rows}))
    (out / "risk.json").write_text(json.dumps({"risk": risk}))

    trajs = build(tracks, H)
    (out / "trajectories.json").write_text(json.dumps([{
        "tid": tr.tid, "group": tr.group, "cls": tr.cls, "t": tr.t.tolist(), "box": tr.box.tolist(), "foot": tr.foot.tolist(),
        "vel": tr.vel.tolist(), "height": tr.height.tolist(), "conf": tr.conf.tolist()} for tr in trajs]))

    sig = json.loads((ROOT / "cache/signal" / f"{args.clip}.json").read_text())
    (out / "signal.json").write_text(json.dumps({k: sig[k] for k in ("times", "raw", "phases", "scores")}))
    ctx = Context(trajs, np.array(sig["times"]), np.array(sig["phases"]), duration)
    events, evidence = E.detect_from_context(ctx, H)
    (out / "rules.json").write_text(json.dumps({
        "dt": ctx.dt, "people": sorted(tr.tid for tr in ctx.people), "events": events,
        "evidence": [{"label": e.label, "start": round(e.start, 4), "end": round(e.end, 4), "actors": e.actors,
                      "note": e.note} for e in evidence]}, indent=1))

    # the fixture clips (C3905, C3902) never trigger red_light, congestion or wrong_way on their own, so the
    # same inputs are perturbed until they do (tests/rules.test.ts applies the same perturbations to the same files)
    import dataclasses

    def flipped(tr):
        return dataclasses.replace(tr, box=tr.box[::-1].copy(), foot=tr.foot[::-1].copy(), vel=-tr.vel[::-1],
                                   height=tr.height[::-1].copy(), conf=tr.conf[::-1].copy())

    phases = np.array(sig["phases"])
    perturbations = {
        "phases_rolled_37": (trajs, np.roll(phases, 37), ["red_light", "congestion"]),
        "slow_on_green": ([dataclasses.replace(tr, vel=tr.vel * 0.3) for tr in trajs], np.full(len(phases), "green"),
                          ["congestion"]),
        "every_10th_reversed": ([flipped(tr) if i % 10 == 0 else tr for i, tr in enumerate(trajs)], phases,
                                ["wrong_way", "illegal_u_turn"]),
    }
    perturbed = {}
    for name, (tr_p, ph_p, labels) in perturbations.items():
        ev_p, evid_p = E.detect_from_context(Context(tr_p, np.array(sig["times"]), ph_p, duration), H)
        counts: dict[str, int] = {}
        for e in evid_p:
            counts[e.label] = counts.get(e.label, 0) + 1
        perturbed[name] = {"labels": labels, "counts": counts,
                           "evidence": [[e.label, e.start, e.end, e.actors, e.note] for e in evid_p if e.label in labels],
                           "events": [x for x in ev_p if x[2] in labels]}
    (out / "rules_perturbed.json").write_text(json.dumps(perturbed, indent=1))

    # alignment input and the lamp reader on real pixels
    video = SAMPLES / f"{args.clip}.MP4"
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
    cv2.imread(str(out / "frame0.png"), cv2.IMREAD_GRAYSCALE).tofile(out / "frame0.gray")
    for item in lamps:
        img = cv2.imread(str(out / item["file"]))
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tofile(out / item["file"].replace(".png", ".rgb"))
        item["size"] = [img.shape[1], img.shape[0]]
    refs = ROOT / "site/tests/fixtures/refs"
    refs.mkdir(exist_ok=True)
    for name in ("reference_day", "reference_dusk"):
        cv2.imread(str(ROOT / "site/public/pipeline" / f"{name}.png"), cv2.IMREAD_GRAYSCALE).tofile(refs / f"{name}.gray")

    # the scene rasters at random points, straight from the Python Context
    rng = np.random.default_rng(0)
    pts = np.round(np.stack([rng.uniform(0, S.REF_SIZE[0] - 1, 3000), rng.uniform(0, S.REF_SIZE[1] - 1, 3000)], axis=1), 2)
    samples = {"points": r(pts, 2)}
    for name in ("road", "walk"):
        samples[name] = ctx.sample(getattr(ctx, name), pts).astype(int).tolist()
    for name in ("road_dist", "walk_dist", "cw_dist"):
        samples[name] = r(ctx.sample(getattr(ctx, name), pts), 3)
    for name, m in ctx.cw_masks.items():
        samples[f"cw_{name}"] = ctx.sample(m, pts).astype(int).tolist()
    for name, m in ctx.zones.items():
        samples[f"zone_{name}"] = ctx.sample(m, pts).astype(int).tolist()
    (out / "scene_samples.json").write_text(json.dumps(samples))

    # the browser detector on one frame, as Ultralytics runs the same ONNX file
    from ultralytics import YOLO

    img = cv2.resize(first, (960, 540), interpolation=cv2.INTER_AREA)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB).tofile(out / "det_frame.rgb")
    onnx = ROOT / "site/public/pipeline/model.onnx"
    res = YOLO(str(onnx), task="detect")(img, imgsz=(544, 960), conf=0.1, verbose=False)[0].boxes
    (out / "det_expected.json").write_text(json.dumps({"size": [960, 540], "boxes": [
        r(list(b) + [c, k]) for b, c, k in zip(res.xyxy.cpu().numpy(), res.conf.cpu().numpy(), res.cls.cpu().numpy())]}))

    (out / "alignment.json").write_text(json.dumps({"H": H.tolist(), "frame": "frame0.png", "frame_size": [480, 270],
                                                    "work_size": [w, h]}))
    (out / "lamps.json").write_text(json.dumps({"boxes": [list(map(int, b)) for b in boxes], "frames": lamps}))
    print(f"{args.clip}: {len(frames)} frames, {len(trajs)} trajectories, {len(events)} events -> {out}")


if __name__ == "__main__":
    main()
