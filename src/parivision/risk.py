"""Part B: causal accident anticipation.

Only frames already received are used. Every third frame (10 Hz) goes through
a small detector and the same ByteTrack thresholds as Part A (with a shorter
1.5 s track buffer); each road user gets a constant-velocity prediction over
the next 3 s, in metres, and every pair that is closing in on each other is
scored by how soon and how deep their predicted footprints overlap. Hard
braking adds to the risk. The score is the worst pair, smoothed with an EMA.

Calibration target: 0.5 should mean "contact is likely within 5 s". On the
sample clips (normal traffic, no collisions) the score stays well below 0.5.
"""
from __future__ import annotations

import os
import time
from functools import lru_cache

import cv2
import numpy as np

from .detector import PERSON, Detector, pick_device
from .pipeline import LAST_RUN, references
from .registration import Alignment, align_best, warp_points
from .scene import masks, metres_per_px
from .tracking import MultiTracker

WORK_WIDTH = 1280
TARGET_HZ = 10.0
MAX_STRIDE_FACTOR = 10  # on a slow machine fall back to 1 Hz, and skip processing entirely if even that is late
# the defaults fit the official T4 run; the environment overrides are for slower machines (our laptop)
OWN_TIME_SHARE = float(os.environ.get("PARIVISION_RISK_SHARE", 0.4))  # our processing, x video time (decode on top)
TOTAL_LIMIT = float(os.environ.get("PARIVISION_TOTAL_LIMIT", 2.8))    # Part A + Part B, x clip duration (harness: 3.0)
HISTORY = 8            # samples used for the velocity estimate (0.8 s)
HORIZON = 3.0          # s of constant-velocity look-ahead
STEP = 0.1
MIN_SPEED = 1.0        # m/s: parked or queued users never trigger
CLOSING_MIN = 4.0      # m/s: slower approaches are ordinary traffic
CLOSING_FULL = 10.0    # m/s: approach speed that gives the full hazard
TTC_HALF = 0.9         # s: contact predicted this soon gives half the maximum hazard
PERSIST_SEC = 0.6      # a pair must look dangerous for this long without a break: at 0.2 s (3 updates) queues
                       # and following cars set off about one alarm per clip at 10 Hz
RADIUS_M = {PERSON: 0.35, 1: 0.6, 3: 0.7, 2: 1.3, 5: 2.0, 7: 1.8}  # footprint radius by COCO class
EMA = 0.35
TRACK_BUFFER_SEC = 1.5  # shorter than Part A's 2 s: a lost pair should stop counting quickly


@lru_cache(maxsize=1)
def _detector() -> Detector:
    if pick_device() == "cpu":
        return Detector("yolo26n.pt", imgsz=640, conf=0.2)
    return Detector("yolo26s.pt", imgsz=960, conf=0.2)


@lru_cache(maxsize=1)
def _road_mask() -> np.ndarray:
    """Carriageway including the crossings: a pedestrian on the pavement is never at risk from traffic."""
    return masks()["road"]


@lru_cache(maxsize=1)
def _carriageways() -> np.ndarray:
    """0 elsewhere, 1 on the southbound carriageway, 2 on the northbound one (the median separates them)."""
    import cv2

    from .rules import DIRECTION_ZONES

    m = np.zeros(_road_mask().shape, np.uint8)
    for code, name in ((1, "sb"), (2, "nb")):
        cv2.fillPoly(m, [np.asarray(DIRECTION_ZONES[name][0], np.int32)], code)
    return m


def pair_hazard(p1, v1, r1, p2, v2, r2) -> float:
    """Hazard in [0, 1] for two road users (positions in m, velocities in m/s, radii in m).

    Constant-velocity closest approach: if the predicted gap between their
    footprints closes within HORIZON s while they approach each other fast,
    the hazard rises with how soon (time of contact) and how deep the
    predicted overlap is. Users moving in parallel (following, adjacent
    lanes) have a small closing speed and score 0.
    """
    rel_p = p2 - p1
    rel_v = v2 - v1
    dist = float(np.linalg.norm(rel_p)) + 1e-6
    closing = -float(np.dot(rel_p, rel_v)) / dist
    if closing < CLOSING_MIN:
        return 0.0
    if min(np.linalg.norm(v1), np.linalg.norm(v2)) < MIN_SPEED:
        # one of them stands still: only a dead-centre course counts (cars pass parked cars all day)
        r1, r2 = 0.4 * r1, 0.4 * r2
    vv = float(np.dot(rel_v, rel_v)) + 1e-6
    t_star = float(np.clip(-np.dot(rel_p, rel_v) / vv, 0.0, HORIZON))
    gap = float(np.linalg.norm(rel_p + t_star * rel_v)) - (r1 + r2)
    if gap >= 0.0:
        return 0.0
    # time at which the footprints first touch
    ts = np.arange(0.0, t_star + 1e-9, STEP)
    touch = np.nonzero(np.linalg.norm(rel_p[None] + ts[:, None] * rel_v[None], axis=1) < r1 + r2)[0]
    ttc = float(ts[touch[0]]) if len(touch) else t_star
    depth = min(1.0, -gap / (r1 + r2))
    soon = 1.0 / (1.0 + np.exp(2.5 * (ttc - TTC_HALF)))   # 0.5 at TTC_HALF seconds
    return float(np.clip(soon * (0.4 + 0.6 * depth) * min(1.0, closing / CLOSING_FULL), 0.0, 1.0))


class Anticipator:
    def __init__(self, meta: dict):
        self.fps = float(meta.get("fps") or 25.0)
        self.stride = max(1, int(round(self.fps / TARGET_HZ)))
        self.width = int(meta.get("width") or 3840)
        self.height = int(meta.get("height") or 2160)
        self.tracker = MultiTracker(fps=self.fps / self.stride, buffer_sec=TRACK_BUFFER_SEC)
        self.alignment: Alignment | None = None
        self.history: dict[int, list[tuple[float, np.ndarray, float, int]]] = {}
        self.base_stride = self.stride
        self.calls = 0
        self.score = 0.0
        self.busy = 0.0  # seconds spent inside step() doing real work
        # wall-clock budget for this whole pass, including the harness decoding frames for us
        duration = float(meta.get("n_frames") or 0) / self.fps if meta.get("n_frames") else 0.0
        spent_a = LAST_RUN.get("seconds", 0.0) if LAST_RUN.get("video") == meta.get("video_id") else 0.0
        self.budget = max(10.0, TOTAL_LIMIT * duration - spent_a) if duration else float("inf")
        self.duration = duration
        self.started: float | None = None
        self.streaks: dict[tuple[int, int], float] = {}

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        k = self.calls
        self.calls += 1
        now = time.perf_counter()
        if self.started is None:
            self.started = now
        if k % self.stride or self._out_of_time(now, t_sec):
            return self.score
        first = self.alignment is None
        self._process(frame, t_sec)
        if not first:  # the first frame loads the model and registers the view: a one-off, not the pace
            self.busy += time.perf_counter() - now
        # stay inside our share of the time budget: thin out frames if we fall behind
        behind = self.busy > OWN_TIME_SHARE * t_sec or self._projected(now, t_sec) > 0.9 * self.budget
        if t_sec > 5.0 and behind:
            self.stride = min(self.base_stride * MAX_STRIDE_FACTOR, self.stride + self.base_stride)
        return self.score

    def _projected(self, now: float, t_sec: float) -> float:
        """Wall time this pass will take at the pace so far (harness decoding included)."""
        if t_sec < 1.0 or not self.duration:
            return 0.0
        return (now - self.started) / t_sec * self.duration

    def _out_of_time(self, now: float, t_sec: float) -> bool:
        """Past 95% of the budget: stop processing and hold the last score to the end."""
        return (now - self.started) > 0.95 * self.budget

    def _process(self, frame: np.ndarray, t_sec: float) -> None:
        step = max(1, int(round(frame.shape[1] / WORK_WIDTH)))
        small = np.ascontiguousarray(frame[::step, ::step])  # 4K -> 1280: plain decimation, ~1 ms
        if self.alignment is None:
            self.alignment = align_best(small, list(references()))
        self.observe(_detector()([small])[0], small.shape[:2], t_sec)

    def observe(self, dets, shape: tuple[int, int], t_sec: float) -> float:
        """Tracker + hazard update from one frame's detections (also used to replay cached detections)."""
        rows = self.tracker.update(dets, shape)
        self._remember(rows, t_sec)
        raw = self._hazard(t_sec)
        self.score = float(np.clip((1 - EMA) * self.score + EMA * raw, 0.0, 1.0))
        return self.score

    def _remember(self, rows: np.ndarray, t: float) -> None:
        live = set()
        if len(rows):
            x1, y1, x2, y2 = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
            foot = warp_points(np.stack([(x1 + x2) / 2, y2], axis=1), self.alignment.H)
            for i, row in enumerate(rows):
                tid, cls = int(row[4]), int(row[6])
                self.history.setdefault(tid, []).append((t, foot[i], RADIUS_M.get(cls, 1.0), cls))
                self.history[tid] = self.history[tid][-HISTORY:]
                live.add(tid)
        for tid in list(self.history):
            if tid not in live and t - self.history[tid][-1][0] > 1.0:
                del self.history[tid]

    def _hazard(self, t: float) -> float:
        users = []
        for tid, h in self.history.items():
            if len(h) < 4 or t - h[-1][0] > 0.25:
                continue
            ts = np.array([x[0] for x in h])
            ps = np.array([x[1] for x in h])  # reference pixels
            v = np.polyfit(ts - ts[-1], ps, 1)[0]  # px/s, least squares: robust to box jitter
            m = metres_per_px(*ps[-1])  # for this user's own speed and braking only
            users.append((ps[-1], v, h[-1][2], h[-1][3], float(np.linalg.norm(v)) * m, self._braking(ts, ps * m), tid))
        best = 0.0
        streak = {}  # pair -> time it started to look dangerous
        for i in range(len(users)):
            q1, v1, r1, c1, s1, b1, k1 = users[i]
            for j in range(i + 1, len(users)):
                q2, v2, r2, c2, s2, b2, k2 = users[j]
                if c1 == PERSON and c2 == PERSON:
                    continue
                # one scale for both users, taken between them: positions scaled by each user's own
                # scale would not share a frame, and users far apart would look close
                m = metres_per_px(*((q1 + q2) / 2))
                p1, p2 = q1 * m, q2 * m
                if max(s1, s2) < MIN_SPEED or np.linalg.norm(p2 - p1) > 30.0:
                    continue
                if min(s1, s2) < MIN_SPEED and PERSON not in (c1, c2):
                    continue  # moving car vs parked or queued car: never on its own
                if (c1 == PERSON and not self._on_road(q1)) or (c2 == PERSON and not self._on_road(q2)):
                    continue
                if {self._lookup(_carriageways(), q1), self._lookup(_carriageways(), q2)} == {1, 2}:
                    continue  # opposite sides of the median
                h = pair_hazard(p1, v1 * m, r1, p2, v2 * m, r2)
                if h <= 0.0:
                    continue
                key = (min(k1, k2), max(k1, k2))
                streak[key] = self.streaks.get(key, t)
                if t - streak[key] >= PERSIST_SEC - 1e-6:
                    best = max(best, min(1.0, h * (1.0 + 0.5 * max(b1, b2))))
        self.streaks = streak
        return best

    def _on_road(self, p: np.ndarray) -> bool:
        return bool(self._lookup(_road_mask(), p))

    @staticmethod
    def _lookup(m: np.ndarray, p: np.ndarray) -> int:
        x, y = int(round(p[0])), int(round(p[1]))
        return int(m[y, x]) if 0 <= x < m.shape[1] and 0 <= y < m.shape[0] else 0

    @staticmethod
    def _braking(ts: np.ndarray, ps: np.ndarray) -> float:
        """0..1: how hard the user decelerated over the history window."""
        if len(ts) < 6:
            return 0.0
        mid = len(ts) // 2
        v_old = np.linalg.norm(ps[mid] - ps[0]) / max(1e-3, ts[mid] - ts[0])
        v_new = np.linalg.norm(ps[-1] - ps[mid]) / max(1e-3, ts[-1] - ts[mid])
        decel = (v_old - v_new) / max(1e-3, (ts[-1] - ts[0]) / 2)
        return float(np.clip((decel - 4.0) / 6.0, 0.0, 1.0)) if v_old > 3.0 else 0.0  # m/s^2: >4 is hard braking
