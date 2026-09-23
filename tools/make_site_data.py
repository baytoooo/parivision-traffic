"""Build the website's data files and media from the official sample run.

    PARIVISION_CACHE_DIR=out/analysis python run_submission.py --videos samples --out predictions_samples.json --team PariVision
    python evaluate.py --pred predictions_samples.json --gt labels/dev_labels.json --json out/metrics.json --per-video
    python tools/eda.py --out out/site_data
    python tools/make_site_data.py --site site/public

Writes data/clips.json, data/results/<clip>.json, data/metrics.json, data/examples.json,
data/eda.json, data/ablations.json, data/runtime.json (with --machine), data/predictions_samples.json
and media/ (annotated videos, posters, example frames, the home-page loop).
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
              "C3905": ("2026-09-18 17:22", "dusk, underexposed, headlights on")}
COUNT_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

HERO_CLIP = "C3896"  # the home-page loop is cut from this clip (clips.json lists it first)

# Zone markers of the dashboard map (site/src/lib/classes.ts ZONES), in percent of the reference view.
ZONES = {"sb_approach": (22, 28), "stop_line": (27, 42), "north_crossing": (47, 56), "median_nose": (64, 49),
         "nb_carriageway": (62, 22), "west_crossing": (25, 83), "junction_box": (66, 76)}
NOTE_ZONE = {"north_sb": ("north_crossing", "north crossing, SB half"),
             "north_nb": ("north_crossing", "north crossing, NB half"),
             "west": ("west_crossing", "west crossing"),
             "NB lane": ("nb_carriageway", "NB lane"), "junction box": ("junction_box", "junction box"),
             "sb": ("sb_approach", "SB carriageway"), "nb": ("nb_carriageway", "NB carriageway")}
CLASS_ZONE = {"red_light": "stop_line", "stop_line": "stop_line", "congestion": "sb_approach",
              "illegal_u_turn": "median_nose"}


def local_start(clip: str) -> str:
    """Recording start in Tashkent time (UTC+5) to the second, from the file's creation time (tools/eda.py)."""
    from datetime import datetime, timedelta

    eda = ROOT / "out/site_data/eda.json"
    meta = next((c for c in json.loads(eda.read_text())["clips"] if c["id"] == clip), None) if eda.exists() else None
    if not meta or not meta.get("created_utc"):
        return LOCAL_TIME.get(clip, ("", ""))[0]
    utc = datetime.fromisoformat(meta["created_utc"].replace("Z", "+00:00"))
    return (utc + timedelta(hours=5)).strftime("%Y-%m-%d %H:%M:%S")


def signal_segments(t: np.ndarray, phase: np.ndarray) -> list:
    segs: list[list] = []
    for ti, p in zip(t.tolist(), phase.tolist()):
        if segs and segs[-1][2] == p:
            segs[-1][1] = round(ti, 1)
        else:
            segs.append([round(ti, 1), round(ti, 1), p])
    return segs


def where(ev, by_id: dict) -> tuple[str, str]:
    """(zone key, readable note) for one piece of evidence."""
    if ev.note in NOTE_ZONE:
        return NOTE_ZONE[ev.note]
    if ev.label in CLASS_ZONE:
        return CLASS_ZONE[ev.label], ev.note or ""
    # otherwise the marker nearest to the first actor's foot at the middle of the event
    tr = by_id.get(ev.actors[0]) if ev.actors else None
    i = tr.at((ev.start + ev.end) / 2) if tr is not None else None
    if i is None:
        return "junction_box", ev.note or ""
    x, y = tr.foot[i] / np.array([19.2, 10.8])
    zone = min(ZONES, key=lambda z: (ZONES[z][0] - x) ** 2 + (ZONES[z][1] - y) ** 2)
    return zone, ev.note or ""


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


def busiest_window(evidence, duration: float, seconds: float) -> float:
    """Start of the window with the most events going on, for the home-page loop."""
    starts = np.arange(0.0, max(0.0, duration - seconds) + 1e-9, 1.0)
    score = [sum(1 for e in evidence if e.start < t0 + seconds and e.end > t0 and e.end - e.start < 60) for t0 in starts]
    return float(starts[int(np.argmax(score))]) if len(starts) else 0.0


def hero_loop(annotated: Path, start: float, seconds: float, out: Path, poster: Path, panel_h: int = 70) -> None:
    """Cut a silent loop from an annotated render, without its timeline panel."""
    import av

    with av.open(str(annotated)) as src, av.open(str(out), "w") as dst:
        vin = src.streams.video[0]
        rate = int(round(float(vin.average_rate)))
        stream = None
        for frame in src.decode(vin):
            t = float(frame.pts * vin.time_base)
            if t < start:
                continue
            if t >= start + seconds:
                break
            img = frame.to_ndarray(format="bgr24")[:-panel_h]
            if stream is None:
                stream = dst.add_stream("libx264", rate=rate)
                stream.width, stream.height = img.shape[1], img.shape[0]
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "28", "preset": "slow", "movflags": "+faststart"}
                cv2.imwrite(str(poster), img, [cv2.IMWRITE_JPEG_QUALITY, 80])
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                dst.mux(packet)
        if stream is not None:
            for packet in stream.encode():
                dst.mux(packet)


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
                        "caption": where(ev, boxes_at)[1]})
            j += 1
        if j >= len(targets):
            break
    return out


def label_stats(labels: dict) -> dict:
    """Events per class in our dev labels: count, clips it occurs in, median and total length."""
    out: dict[str, dict] = {}
    for key, v in labels.items():
        for s, e, lab in v["events"]:
            d = out.setdefault(lab, {"n": 0, "clips": set(), "lengths": []})
            d["n"] += 1
            d["clips"].add(Path(key).stem)
            d["lengths"].append(e - s)
    return {lab: {"n": d["n"], "clips": sorted(d["clips"]), "median_sec": round(float(np.median(d["lengths"])), 1),
                  "total_sec": round(float(np.sum(d["lengths"])), 1)} for lab, d in sorted(out.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(ROOT / "site/public"))
    ap.add_argument("--analysis", default=str(ROOT / "out/analysis"))
    ap.add_argument("--samples", default=str(ROOT / "kit/samples"))
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--machine", default="", help="where predictions_samples.json was produced, for runtime.json")
    ap.add_argument("--runtime-note", default="")
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
        by_id = {tr.tid: tr for tr in a.trajectories}
        light = LOCAL_TIME.get(clip, ("", ""))[1]
        local = local_start(clip)
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
                          "zone": z, "note": n} for e in a.evidence for z, n in [where(e, by_id)]],
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
            if clip == HERO_CLIP:
                t0 = busiest_window(a.evidence, a.info.duration, 20.0)
                hero_loop(media / f"{clip}_annotated.mp4", t0, 20.0, media / "loop.mp4", media / "loop_poster.jpg")
        print(clip, len(result["events"]), "events,", len(result["labels"]), "labels")

    (data / "clips.json").write_text(json.dumps(clips))
    if examples:
        (data / "examples.json").write_text(json.dumps(examples))
    shutil.copy(ROOT / "predictions_samples.json", data / "predictions_samples.json")
    if args.machine and preds.get("log"):
        rows = [{"clip": Path(k).stem, "duration": v["duration"], "part_a_sec": v.get("part_a_sec"),
                 "part_b_sec": v.get("part_b_sec"), "total_sec": v.get("total_sec")} for k, v in preds["log"].items()]
        (data / "runtime.json").write_text(json.dumps({"machine": args.machine, "note": args.runtime_note,
                                                       "clips": rows}, indent=1))
    ablations = ROOT / "out/ablations.json"
    if ablations.exists():
        shutil.copy(ablations, data / "ablations.json")
    metrics = ROOT / "out/metrics.json"
    if metrics.exists():
        shutil.copy(metrics, data / "metrics.json")
    eda = ROOT / "out/site_data/eda.json"
    if eda.exists():
        e = json.loads(eda.read_text())
        e["labels"] = label_stats(labels)
        (data / "eda.json").write_text(json.dumps(e))
        for img in (ROOT / "out/site_data/media/eda").glob("*.jpg"):
            shutil.copy(img, media / "eda" / img.name)


if __name__ == "__main__":
    main()
