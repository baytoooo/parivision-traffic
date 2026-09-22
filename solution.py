"""
solution.py — team PariVision.

Part A (detect_events): YOLO26 detector + ByteTrack on every 3rd frame, tracks
mapped into a reference view of the junction, then hand-written rules on the
trajectories and the traffic-signal phase. See README.md and docs/.

Part B (RiskEstimator): causal; a lighter detector + tracker on every 3rd frame
it receives, time-to-collision between road users mapped to a probability.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("YOLO_OFFLINE", "1")  # the evaluation machine has no internet; skip ultralytics' online checks
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from parivision.pipeline import analyse  # noqa: E402
from parivision.risk import Anticipator  # noqa: E402
from parivision.seed import fix_seeds  # noqa: E402

CLASSES: list[str] = [
    "accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
    "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
    "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke",
]

RISK_HORIZON_SEC = 5.0

fix_seeds()


def detect_events(video_path: str) -> list[list]:
    return analyse(video_path).events


class RiskEstimator:
    def reset(self, meta: dict) -> None:
        self.model = Anticipator(meta)

    def step(self, frame: np.ndarray, t_sec: float) -> float:
        return self.model.step(frame, t_sec)
