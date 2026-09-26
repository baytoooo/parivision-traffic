"""The rules on one clip's saved trajectories still give the submitted events.

tests/data/C3905_trajectories.npz holds what Part A had built on C3905 when predictions_samples.json
was made (tracks in reference pixels, the signal timeline, the registration), written from the
PARIVISION_CACHE_DIR analysis. Rerunning the detector takes an hour of 4K decoding; rerunning the
rules takes seconds, so this catches any rule change that would alter the committed predictions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parivision.events import detect_from_context  # noqa: E402
from parivision.rules import Context  # noqa: E402
from parivision.trajectories import Trajectory  # noqa: E402


def _load(path: Path) -> tuple[Context, np.ndarray]:
    z = np.load(path)
    ends = np.cumsum(z["n"])
    trajectories = []
    for k, (a, b) in enumerate(zip(ends - z["n"], ends)):
        trajectories.append(Trajectory(int(z["tid"][k]), str(z["group"][k]), int(z["cls"][k]), z["t"][a:b], z["box"][a:b],
                                       z["foot"][a:b], z["vel"][a:b], z["height"][a:b], z["conf"][a:b]))
    ctx = Context(trajectories, z["signal_t"], z["signal_phase"].astype(object), float(z["duration"]))
    return ctx, z["H"]


def test_rules_reproduce_the_submitted_events_on_c3905():
    ctx, H = _load(ROOT / "tests/data/C3905_trajectories.npz")
    events, _ = detect_from_context(ctx, H)
    submitted = json.loads((ROOT / "predictions_samples.json").read_text())["videos"]["C3905.MP4"]["events"]
    # the harness rounds times to 3 decimals (run_submission.clean_events)
    assert sorted([round(s, 3), round(e, 3), label] for s, e, label in events) == sorted(submitted)
