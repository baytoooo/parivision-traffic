"""Multi-object tracking on top of per-frame detections.

ByteTrack (the Ultralytics implementation) associates boxes by IoU only, so
people and vehicles get separate tracker instances: a pedestrian on a
crossing would otherwise be glued to the car passing in front of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from .detector import ANIMALS, BICYCLE, PERSON, VEHICLES, Detections

GROUPS = {
    "vehicle": tuple(VEHICLES),
    "person": (PERSON,),
    "bicycle": (BICYCLE,),
    "animal": tuple(ANIMALS),
}


def _byte_args(fps: float, buffer_sec: float) -> SimpleNamespace:
    return SimpleNamespace(
        tracker_type="bytetrack",
        track_high_thresh=0.3,
        track_low_thresh=0.1,
        new_track_thresh=0.4,
        track_buffer=max(1, int(round(buffer_sec * fps))),
        match_thresh=0.8,
        fuse_score=True,
    )


class MultiTracker:
    """One ByteTrack per object group; ids are unique across groups."""

    def __init__(self, fps: float, buffer_sec: float = 2.0):
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.trackers = {g: BYTETracker(_byte_args(fps, buffer_sec)) for g in GROUPS}
        # BYTETracker keeps a global id counter; offset ids per group so they never collide.
        self.offsets = {g: (i + 1) * 1_000_000 for i, g in enumerate(GROUPS)}

    def update(self, det: Detections, shape: tuple[int, int]) -> np.ndarray:
        """Returns rows ``[x1, y1, x2, y2, track_id, score, cls]`` for active tracks."""
        from ultralytics.engine.results import Boxes

        rows = []
        for group, classes in GROUPS.items():
            mask = np.isin(det.cls, classes)
            trk = self.trackers[group]
            if not mask.any() and not trk.tracked_stracks and not trk.lost_stracks:
                trk.frame_id += 1  # nothing to associate; just keep the frame clock in step
                continue
            data = np.concatenate(
                [det.xyxy[mask], det.conf[mask, None], det.cls[mask, None].astype(np.float32)], axis=1
            ) if mask.any() else np.zeros((0, 6), np.float32)
            out = trk.update(Boxes(data, shape))
            if len(out):
                out = out[:, :7].copy()
                out[:, 4] += self.offsets[group]
                rows.append(out)
        return np.concatenate(rows) if rows else np.zeros((0, 7), np.float32)


@dataclass
class Track:
    tid: int
    group: str
    t: list[float] = field(default_factory=list)
    box: list[np.ndarray] = field(default_factory=list)  # xyxy in working-frame pixels
    conf: list[float] = field(default_factory=list)
    cls: list[int] = field(default_factory=list)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.t), np.asarray(self.box)

    @property
    def label(self) -> int:
        """Majority COCO class over the track's life (cars flicker to truck and back)."""
        vals, counts = np.unique(np.asarray(self.cls), return_counts=True)
        return int(vals[np.argmax(counts)])

    @property
    def duration(self) -> float:
        return self.t[-1] - self.t[0] if self.t else 0.0


def group_of(track_id: float) -> str:
    return list(GROUPS)[int(track_id) // 1_000_000 - 1]


def collect(tracks: dict[int, Track], t: float, rows: np.ndarray) -> None:
    for x1, y1, x2, y2, tid, score, cls in rows:
        tid = int(tid)
        tr = tracks.get(tid)
        if tr is None:
            tr = tracks[tid] = Track(tid, group_of(tid))
        tr.t.append(float(t))
        tr.box.append(np.array([x1, y1, x2, y2], np.float32))
        tr.conf.append(float(score))
        tr.cls.append(int(cls))
