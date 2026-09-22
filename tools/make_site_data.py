"""Build the website's data files and media from the official sample run.

    PARIVISION_CACHE_DIR=out/analysis python run_submission.py --videos samples --out predictions_samples.json --team PariVision
    python evaluate.py --pred predictions_samples.json --gt labels/dev_labels.json --json out/metrics.json --per-video
    python tools/eda.py --out out/site_data
    python tools/make_site_data.py --site site/public

Writes data/clips.json, data/results/<clip>.json, data/metrics.json, data/examples.json,
data/eda.json, data/predictions_samples.json and media/ (annotated videos, posters, example frames).
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parivision.render import render  # noqa: E402
from parivision.video import sample_frames  # noqa: E402

LOCAL_TIME = {"C3896": ("2026-09-18 11:18", "midday sun, hard shadows"),
              "C3897": ("2026-09-18 11:24", "midday sun, hard shadows"),
              "C3902": ("2026-09-18 16:58", "low sun, long shadows"),
              "C3905": ("2026-09-18 17:22", "dusk, headlights on")}
COUNT_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def signal_segments(t: np.ndarray, phase: np.ndarray) -> list:
    segs: list[list] = []
    for ti, p in zip(t.tolist(), phase.tolist()):
        if segs and segs[-1][2] == p:
            segs[-1][1] = round(ti, 1)
        else:
            segs.append([round(ti, 1), round(ti, 1), p])
    return segs


def counts(a, bin_s: float = 5.0) -> dict:
    """Distinct tracked road users per 5 s bin, by class."""
    n = int(np.ceil(a.info.duration / bin_s))
    out = {"t": [round(i * bin_s, 1) for i in range(n)]}
    for name in COUNT_NAMES.values():
        out[name] = [0] * n
    for tr in a.trajectories:
        name = COUNT_NAMES.get(tr.cls)
        if name is None:
            continue
        for b in np.unique((tr.t // bin_s).astype(int)):
            if 0 <= b < n:
                out[name][b] += 1
    return out


def example_frames(clip: str, video: Path, a, media: Path, per_class: int = 3) -> list[dict]:
    """A frame from the middle of a few events of each class, with the actors boxed."""
    out = []
    by_label: dict[str, list] = {}
    for ev in a.evidence:
        by_label.setdefault(ev.label, []).append(ev)
    wanted = {}
    for label, evs in by_label.items():
        evs = sorted(evs, key=lambda e: -(e.end - e.start))[:per_class]
        for k, ev in enumerate(evs):
            wanted[round((ev.start + ev.end) / 2, 1)] = (label, k, ev)
    if not wanted:
        return out
    boxes_at = {}
    for tr in a.trajectories:
        boxes_at[tr.tid] = tr
    targets = sorted(wanted)
    j = 0
    for _, t, frame in sample_frames(video, 10.0, a.work_size[0]):
        while j < len(targets) and t >= targets[j] - 0.05:
            label, k, ev = wanted[targets[j]]
            img = frame.copy()
            for tid in ev.actors:
                tr = boxes_at.get(tid)
                i = tr.at(t) if tr is not None else None
                if i is not None:
                    x1, y1, x2, y2 = tr.box[i].astype(int)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (40, 200, 255), 3)
            small = cv2.resize(img, (960, int(round(img.shape[0] * 960 / img.shape[1]))), interpolation=cv2.INTER_AREA)
            name = f"{clip}_{label}_{k}.jpg"
            cv2.imwrite(str(media / "examples" / name), small, [cv2.IMWRITE_JPEG_QUALITY, 82])
            out.append({"label": label, "clip": clip, "t": round(ev.start, 1), "thumb": f"/media/examples/{name}",
                        "caption": ev.note or ""})
            j += 1
        if j >= len(targets):
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(ROOT / "site/public"))
    ap.add_argument("--analysis", default=str(ROOT / "out/analysis"))
    ap.add_argument("--samples", default=str(ROOT / "kit/samples"))
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    site = Path(args.site)
    data, media = site / "data", site / "media"
    for d in (data / "results", media / "examples", media / "eda"):
        d.mkdir(parents=True, exist_ok=True)

    preds = json.loads((ROOT / "predictions_samples.json").read_text())
    labels = json.loads((ROOT / "labels/dev_labels.json").read_text())
    clips, examples = [], []
    for pkl in sorted(Path(args.analysis).glob("*.pkl")):
        a = pickle.load(open(pkl, "rb"))
        clip = pkl.stem
        key = next(k for k in preds["videos"] if Path(k).stem == clip)
        video = Path(args.samples) / key
        risk = preds["videos"][key]["risk"]
        risk10 = [r for i, r in enumerate(risk) if i % 3 == 0]  # 29.97 fps -> ~10 Hz for the page
        local, light = LOCAL_TIME.get(clip, ("", ""))
        clips.append({"id": clip, "duration": round(a.info.duration, 1), "fps": round(a.info.fps, 2),
                      "width": a.info.width, "height": a.info.height, "local_time": local, "light": light,
                      "video": f"/media/{clip}_annotated.mp4", "poster": f"/media/{clip}_poster.jpg"})
        result = {
            "clip": clip, "duration": round(a.info.duration, 2),
            "events": preds["videos"][key]["events"],
            "labels": labels.get(key, {}).get("events", []),
            "risk": risk10,
            "signal": signal_segments(a.signal_t, a.signal_phase),
            "evidence": [{"label": e.label, "start": round(e.start, 2), "end": round(e.end, 2), "actors": e.actors,
                          "note": e.note} for e in a.evidence],
            "counts": counts(a),
        }
        (data / "results" / f"{clip}.json").write_text(json.dumps(result))
        if not args.no_video:
            render(str(video), str(media / f"{clip}_annotated.mp4"), a.trajectories, a.events, a.evidence,
                   a.signal_t, a.signal_phase, a.alignment.H, a.work_size[0], risk=risk10, out_width=960, fps=10.0,
                   duration=a.info.duration)
            cap = cv2.VideoCapture(str(media / f"{clip}_annotated.mp4"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
            ok, poster = cap.read()
            if ok:
                cv2.imwrite(str(media / f"{clip}_poster.jpg"), poster, [cv2.IMWRITE_JPEG_QUALITY, 80])
            examples += example_frames(clip, video, a, media)
        print(clip, len(result["events"]), "events,", len(result["labels"]), "labels")

    (data / "clips.json").write_text(json.dumps(clips))
    if examples:
        (data / "examples.json").write_text(json.dumps(examples))
    shutil.copy(ROOT / "predictions_samples.json", data / "predictions_samples.json")
    metrics = ROOT / "out/metrics.json"
    if metrics.exists():
        shutil.copy(metrics, data / "metrics.json")
    eda = ROOT / "out/site_data/eda.json"
    if eda.exists():
        shutil.copy(eda, data / "eda.json")
        for img in (ROOT / "out/site_data/media/eda").glob("*.jpg"):
            shutil.copy(img, media / "eda" / img.name)


if __name__ == "__main__":
    main()
