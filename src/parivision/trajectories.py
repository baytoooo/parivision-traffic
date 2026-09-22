"""Tracks as smoothed trajectories in reference coordinates.

Rules reason about where a road user stands on the ground, so every box is
reduced to its foot point (bottom centre; for vehicles a bit above the
bottom edge, which sits closer to the middle of the footprint in this
oblique view) and mapped into the reference frame. Velocities come from a
centred finite difference on the smoothed track.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

from .detector import BUS, CAR, MOTORCYCLE, PERSON, TRUCK
from .registration import warp_points
from .tracking import Track


@dataclass
class Trajectory:
    tid: int
    group: str            # vehicle / person / bicycle / animal
    cls: int              # majority COCO class
    t: np.ndarray         # (N,) seconds
    box: np.ndarray       # (N, 4) xyxy in working-frame pixels
    foot: np.ndarray      # (N, 2) reference pixels, smoothed
    vel: np.ndarray       # (N, 2) reference pixels per second
    height: np.ndarray    # (N,) box height in reference pixels (a rough scale for "how far is 1 m here")
    conf: np.ndarray      # (N,)

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.vel, axis=1)

    @property
    def is_vehicle(self) -> bool:
        return self.cls in (CAR, BUS, TRUCK, MOTORCYCLE)

    @property
    def is_person(self) -> bool:
        return self.cls == PERSON

    def at(self, t: float) -> int | None:
        """Index of the sample closest to ``t`` if the track exists then."""
        if t < self.t[0] - 1e-6 or t > self.t[-1] + 1e-6:
            return None
        return int(np.argmin(np.abs(self.t - t)))


def foot_points(box: np.ndarray, is_vehicle: bool) -> np.ndarray:
    x1, y1, x2, y2 = box.T
    cx = (x1 + x2) / 2
    # vehicle boxes include the roof; the ground contact is near the bottom but the
    # footprint centre is a little higher. Persons stand on the bottom edge.
    fy = y2 - (0.15 * (y2 - y1) if is_vehicle else 0.0)
    return np.stack([cx, fy], axis=1)


def build(tracks: dict[int, Track], H: np.ndarray, min_len: int = 5, smooth: int = 5) -> list[Trajectory]:
    """Turn raw tracker output into trajectories. ``H`` maps working pixels to reference pixels."""
    out = []
    for tr in tracks.values():
        if len(tr.t) < min_len:
            continue
        t, box = tr.arrays()
        order = np.argsort(t)
        t, box = t[order], box[order]
        cls = tr.label
        is_vehicle = cls in (CAR, BUS, TRUCK, MOTORCYCLE)
        foot = warp_points(foot_points(box, is_vehicle), H)
        top = warp_points(np.stack([(box[:, 0] + box[:, 2]) / 2, box[:, 1]], axis=1), H)
        height = np.linalg.norm(foot - top, axis=1)
        k = min(smooth, len(t))
        foot_s = uniform_filter1d(foot, size=k, axis=0, mode="nearest")
        vel = np.gradient(foot_s, t, axis=0) if len(t) > 1 else np.zeros_like(foot_s)
        out.append(Trajectory(tr.tid, tr.group, cls, t, box, foot_s, vel, height,
                              np.asarray(tr.conf, np.float32)[order]))
    return out
