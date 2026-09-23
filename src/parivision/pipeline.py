"""Part A end to end: one clip in, events (and everything needed to draw them) out."""
from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from . import scene as S
from .detector import Detector, pick_device
from .events import detect_from_context
from .registration import Alignment, align_best
from .rules import Context, Evidence
from .signal import fill_phases, lamp_patches, lamp_scores, phase_from_scores
from .tracking import MultiTracker, collect
from .trajectories import Trajectory, build
from .video import VideoInfo, probe, sample_frames

WORK_WIDTH = 1920       # frames are decoded straight to this width (reference view resolution)
BATCH = 8
TIME_SHARE = float(os.environ.get("PARIVISION_TIME_SHARE", 1.3))  # Part A's share of the 3x budget

# Wall time Part A spent on the last clip. The harness runs Part B on the same clip
# right after Part A in the same process, so the risk model can budget the rest of
# the 3x limit. This is timing only: no Part A result ever reaches Part B.
LAST_RUN: dict = {}
MAX_THIN = 3            # on a slow machine analyse at most every 3rd sampled frame (10 -> 3.3 fps)
ASSETS = Path(__file__).resolve().parent / "assets"

# (weights, detector input size, analysed frames per second). The GPU profile is
# what we tuned and submit; the CPU one only exists so a machine without a
# working CUDA driver still finishes inside the time budget.
PROFILES = {
    "gpu": ("yolo26m.pt", 1280, 10.0),
    "cpu": ("yolo26s.pt", 960, 5.0),
}


def profile() -> tuple[str, int, float]:
    return PROFILES["cpu" if pick_device() == "cpu" else "gpu"]


@dataclass
class Analysis:
    info: VideoInfo
    events: list[list]
    evidence: list[Evidence]
    trajectories: list[Trajectory]
    signal_t: np.ndarray
    signal_phase: np.ndarray
    alignment: Alignment
    work_size: tuple[int, int]
    processed_until: float
    seconds: float


@lru_cache(maxsize=1)
def detector() -> Detector:
    weights, imgsz, _ = profile()
    return Detector(weights, imgsz=imgsz, conf=0.1)


@lru_cache(maxsize=1)
def references() -> tuple[np.ndarray, ...]:
    """Median backgrounds of the junction (midday and dusk), both in reference-view pixels."""
    imgs = tuple(cv2.imread(str(p)) for p in sorted(ASSETS.glob("reference_*.jpg")))
    if not imgs or any(i is None for i in imgs):
        raise FileNotFoundError(f"reference images missing in {ASSETS}")
    return imgs


def analyse(video_path: str, time_share: float = TIME_SHARE, progress=None,
            max_seconds: float | None = None, on_frame=None) -> Analysis:
    """Detect events in one clip.

    ``progress(stage, fraction)`` is called as the clip is processed (used by
    the web demo), ``max_seconds`` stops early (demo upload limit), and
    ``on_frame(t, detections, frame, H)`` sees every analysed frame's
    detections in time order (the demo feeds them to the causal risk model
    instead of running a second detector).
    """
    t_start = time.perf_counter()
    info = probe(video_path)
    limit = min(info.duration, max_seconds) if max_seconds else info.duration
    deadline = t_start + max(60.0, time_share * info.duration)
    det = detector()
    sample_fps = profile()[2]
    tracker = MultiTracker(fps=sample_fps)
    tracks: dict = {}
    alignment: Alignment | None = None
    boxes = None
    sig_t: list[float] = []
    sig_raw: list[str] = []
    pending: list[tuple[float, np.ndarray]] = []
    last_t = 0.0
    thin, n_seen = 1, 0  # analyse every `thin`-th sampled frame; raised if we would miss the deadline
    mark = (0.0, t_start)  # (video time, wall time) at the last pace check

    def flush() -> None:
        results = det([img for _, img in pending])
        for (t, img), d in zip(pending, results):
            collect(tracks, t, tracker.update(d, img.shape[:2]))
            if on_frame is not None:
                on_frame(t, d, img, alignment.H)
        if progress is not None and pending:
            progress("detecting and tracking", min(1.0, pending[-1][0] / max(limit, 1e-6)))
        pending.clear()

    if progress is not None:
        progress("aligning the view", 0.0)
    for _, t, img in sample_frames(video_path, sample_fps, WORK_WIDTH):
        if t > limit:
            break
        if alignment is None:
            alignment = align_best(img, list(references()))
            boxes = lamp_patches(np.linalg.inv(alignment.H))
        sig_t.append(t)
        sig_raw.append(phase_from_scores(lamp_scores(img, boxes)))
        n_seen += 1
        if n_seen % thin:
            continue
        pending.append((t, img))
        last_t = t
        if len(pending) == BATCH:
            flush()
            now = time.perf_counter()
            if now > deadline:
                break
            # projected finish at the recent pace; thin out frames rather than stop early
            if t - mark[0] >= 10.0:
                pace = (now - mark[1]) / (t - mark[0])  # wall seconds per video second
                if thin < MAX_THIN and now + pace * (limit - t) > t_start + 0.95 * (deadline - t_start):
                    thin += 1
                mark = (t, now)
    if pending:
        flush()

    work_h = int(round(info.height * WORK_WIDTH / info.width))
    if alignment is None:  # unreadable clip
        alignment = Alignment(np.diag([S.REF_SIZE[0] / WORK_WIDTH, S.REF_SIZE[1] / work_h, 1.0]), 0, False)
    if progress is not None:
        progress("applying event rules", 1.0)
    trajectories = build(tracks, alignment.H)
    phases = fill_phases(sig_raw, np.asarray(sig_t))
    ctx = Context(trajectories, np.asarray(sig_t), np.asarray(phases), limit)
    events, evidence = detect_from_context(ctx, alignment.H)
    result = Analysis(info, events, evidence, trajectories, np.asarray(sig_t), np.asarray(phases),
                      alignment, (WORK_WIDTH, work_h), last_t, time.perf_counter() - t_start)
    LAST_RUN.update(video=Path(video_path).name, seconds=result.seconds, duration=info.duration)
    _keep(result)
    return result


def _keep(result: Analysis) -> None:
    """Save the full analysis when PARIVISION_CACHE_DIR is set (the website renders from it).

    Off by default: the official run writes nothing besides predictions.json.
    """
    out = os.environ.get("PARIVISION_CACHE_DIR")
    if not out:
        return
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / f"{Path(result.info.path).stem}.pkl", "wb") as fh:
        pickle.dump(result, fh)
