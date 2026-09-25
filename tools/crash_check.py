"""Check the accident rule and the risk model on public CCTV crash clips.

    python tools/crash_check.py fetch --set real        # download the clips listed in labels/accident_real.csv
    python tools/crash_check.py cache --set real        # Part A and Part B detections for them
    python tools/crash_check.py eval --set real --ours  # rule recall and Score B (--ours: our samples as negatives)

--set synthetic does the same for the benchmark's CARLA clips (labels/accident_synthetic.csv).

The clips are real crashes filmed by fixed traffic cameras, from the ACCIDENT benchmark
(CVPR 2026, https://github.com/accidentbench/ACCIDENT, Kaggle picekl/accident, CC BY-NC-SA 4.0;
annotations CC BY 4.0). They are not redistributed here: `fetch` downloads the selection with
kagglehub (pip install kagglehub; not needed by the submission). Each clip comes from another camera, so nothing drawn for our junction applies: the
view is not registered, the whole frame counts as road, and metres per pixel come from the size of
the vehicles in the clip, calibrated against our own scale map (vehicle_scale).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from evaluate import evaluate_part_b  # noqa: E402

from parivision import risk as R  # noqa: E402
from parivision import rules  # noqa: E402
from parivision.detector import Detections, Detector  # noqa: E402
from parivision.registration import Alignment  # noqa: E402
from parivision.scene import metres_per_px  # noqa: E402
from parivision.tracking import MultiTracker, collect  # noqa: E402
from parivision.trajectories import build  # noqa: E402
from parivision.video import sample_frames  # noqa: E402

EXT = ROOT / "external/accident"
CACHE = ROOT / "cache/ext"
FPS = 10.0
WORK = 1920          # Part A work width, as in pipeline.py
SMALL = 1280         # Part B work width, as in risk.py
VEHICLES = (2, 3, 5, 7)


def fetch(clips: pd.DataFrame) -> None:
    import shutil
    from concurrent.futures import ThreadPoolExecutor

    import kagglehub

    (EXT / "videos").mkdir(parents=True, exist_ok=True)

    def get(path: str) -> None:
        dst = EXT / "videos" / Path(path).name
        if not dst.exists():
            shutil.move(kagglehub.dataset_download("picekl/accident", path=path), dst)

    with ThreadPoolExecutor(6) as ex:
        list(ex.map(get, clips.path))


def cache(clips: pd.DataFrame) -> None:
    """Part A detections (YOLO26m @1280 on 1920 px frames) and Part B detections (YOLO26s @960 on 1280 px)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    det_a = Detector("yolo26m.pt", imgsz=1280, conf=0.1)
    det_b = Detector("yolo26s.pt", imgsz=960, conf=0.2)
    for path in clips.path:
        out = CACHE / (Path(path).stem + ".npz")
        video = EXT / "videos" / Path(path).name
        if out.exists() or not video.exists():
            continue
        rows_a, rows_b, batch, height = [], [], [], 0
        def flush():
            imgs = [img for _, _, img in batch]
            small = [img[::3, ::3] if img.shape[1] >= 3 * SMALL else _resize(img, SMALL) for img in imgs]
            for (k, t, _), da, db in zip(batch, det_a(imgs), det_b(small)):
                rows_a.extend([k, t, *b, c, cl] for b, c, cl in zip(da.xyxy, da.conf, da.cls))
                rows_b.extend([k, t, *b, c, cl] for b, c, cl in zip(db.xyxy, db.conf, db.cls))
            batch.clear()
        for k, t, img in sample_frames(video, FPS, WORK):
            height = img.shape[0]
            batch.append((k, t, img))
            if len(batch) == 8:
                flush()
        if batch:
            flush()
        np.savez_compressed(out, a=np.array(rows_a, np.float32).reshape(-1, 8), b=np.array(rows_b, np.float32).reshape(-1, 8),
                            height=height)
        print(Path(path).stem, len(rows_a), "A boxes", len(rows_b), "B boxes", flush=True)


def _resize(img: np.ndarray, width: int) -> np.ndarray:
    import cv2

    return cv2.resize(img, (width, int(round(img.shape[0] * width / img.shape[1]))), interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=1)
def vehicle_metres() -> float:
    """sqrt(box area) of a car in metres, from our own clips where the scale map is known."""
    a = pickle.load(open(ROOT / "out/analysis/C3896.pkl", "rb"))
    vals = []
    for tr in a.trajectories:
        if tr.cls != 2:
            continue
        w, h = tr.box[:, 2] - tr.box[:, 0], tr.box[:, 3] - tr.box[:, 1]
        m = np.array([metres_per_px(*p) for p in tr.foot])
        vals.append(np.sqrt(w * h) * m)
    return float(np.median(np.concatenate(vals)))


def vehicle_scale(rows: np.ndarray):
    """metres per pixel at (x, y) for one clip: a plane fitted to the size of the confident vehicle boxes."""
    v = rows[(rows[:, 6] > 0.4) & np.isin(rows[:, 7], VEHICLES)]
    if len(v) < 20:
        return lambda x, y: vehicle_metres() / 60.0
    x, y = (v[:, 2] + v[:, 4]) / 2, v[:, 5]
    size = np.sqrt((v[:, 4] - v[:, 2]) * (v[:, 5] - v[:, 3]))
    A = np.stack([np.ones_like(x), x, y], axis=1)
    coef = np.linalg.lstsq(A, size, rcond=None)[0]
    lo = np.percentile(size, 5)
    return lambda px, py: vehicle_metres() / max(lo, coef[0] + coef[1] * px + coef[2] * py)


def detections(rows: np.ndarray, k: int) -> Detections:
    r = rows[rows[:, 0] == k]
    return Detections(r[:, 2:6].copy(), r[:, 6].copy(), r[:, 7].astype(np.int16))


def replay(stem: str, duration: float) -> tuple[list, list]:
    """Accident evidence (Part A rule) and the risk curve (Part B) for one cached clip."""
    z = np.load(CACHE / f"{stem}.npz")
    a, b, height = z["a"], z["b"], int(z["height"])
    mpp = vehicle_scale(a)
    frames = np.unique(np.concatenate([a[:, 0], b[:, 0]])).astype(int) if len(a) + len(b) else np.array([], int)
    times = {int(k): float(t) for k, t in np.concatenate([a[:, :2], b[:, :2]])} if len(frames) else {}

    tracker, tracks = MultiTracker(fps=FPS), {}
    for k in frames:
        collect(tracks, times[k], tracker.update(detections(a, k), (height, WORK)))
    trajectories = build(tracks, np.eye(3))
    evidence = rules.collisions(trajectories, mpp)

    R.metres_per_px = mpp                                    # no scale map for this camera
    R._carriageways = lambda: np.zeros((1, 1), np.uint8)     # no median to separate the directions
    ant = R.Anticipator({"fps": FPS, "n_frames": 0})
    ant._on_road = lambda p: True
    ant.alignment = Alignment(np.diag([WORK / SMALL, WORK / SMALL, 1.0]), 0, True)
    small_h = int(round(height * SMALL / WORK))
    curve = [[times[k], ant.observe(detections(b, k), (small_h, SMALL), times[k])] for k in frames]
    return evidence, curve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["fetch", "cache", "eval"])
    ap.add_argument("--set", default="real", choices=["real", "synthetic"])
    ap.add_argument("--ours", action="store_true", help="add our four samples as crash-free negatives")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    clips = pd.read_csv(ROOT / "labels" / f"accident_{args.set}.csv")
    if args.what == "fetch":
        fetch(clips)
        return
    if args.what == "cache":
        cache(clips)
        return

    gt, pred, hits, rows = {}, {}, 0, []
    for c in clips.itertuples():
        stem = Path(c.path).stem
        if not (CACHE / f"{stem}.npz").exists():
            continue
        s = float(c.accident_time)
        evidence, curve = replay(stem, float(c.duration))
        found = [e for e in evidence if s - 1.5 <= e.start <= s + 2.0]
        hits += bool(found)
        early = [e for e in evidence if e.start < s - 1.5]
        gt[stem] = {"events": [[s, min(float(c.duration), s + 5.0), "accident"]]}
        pred[stem] = {"risk": curve}
        peak = max([r for t, r in curve if s - 5 <= t < s], default=0.0)
        rows.append({"clip": stem, "type": c.type, "impact": s, "found": bool(found), "early": len(early),
                     "risk_peak_before": round(peak, 3)})
    if args.ours:
        for p in sorted((ROOT / "out/analysis").glob("*.pkl")):
            key = p.stem + ".MP4"
            full = json.loads((ROOT / "predictions_samples.json").read_text())["videos"][key]
            gt[p.stem] = {"events": []}
            pred[p.stem] = {"risk": full["risk"]}
    n = len(rows)
    b = evaluate_part_b(gt, pred) or {}
    report = {"clips": n, "rule_recall": round(hits / max(1, n), 3),
              "rule_early_clips": sum(r["early"] > 0 for r in rows),
              "score_b": round(b.get("score_b", 0.0), 3), "ap": round(b.get("ap", 0.0), 3),
              "f1_alarm": round(b.get("f1_alarm", 0.0), 3), "mtta_sec": round(b.get("mtta_sec", 0.0), 2),
              "alarms": b.get("n_alarms"), "matched": b.get("n_matched"), "per_clip": rows}
    Path(args.out or ROOT / f"out/crash_check_{args.set}.json").write_text(json.dumps(report, indent=1))
    print({k: v for k, v in report.items() if k != "per_clip"})


if __name__ == "__main__":
    main()
