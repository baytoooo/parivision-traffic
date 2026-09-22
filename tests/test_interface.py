"""The harness contract: solution.py imports, returns valid events and a causal risk curve.

Uses a 6 s synthetic clip (the reference background with a box sliding across),
so it runs anywhere in under a minute. Run: pytest -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def solution():
    spec = importlib.util.spec_from_file_location("solution", ROOT / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["solution"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    bg = cv2.imread(str(ROOT / "src/parivision/assets/reference_day.jpg"))
    path = tmp_path_factory.mktemp("clip") / "synthetic.mp4"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (bg.shape[1], bg.shape[0]))
    for i in range(150):
        f = bg.copy()
        x = 300 + i * 8
        cv2.rectangle(f, (x, 700), (x + 160, 790), (40, 40, 40), -1)
        out.write(f)
    out.release()
    return path


def test_classes_are_official(solution):
    sys.path.insert(0, str(ROOT))
    from evaluate import OFFICIAL_CLASSES

    assert set(solution.CLASSES) <= set(OFFICIAL_CLASSES)


def test_detect_events_returns_valid_segments(solution, clip):
    events = solution.detect_events(str(clip))
    assert isinstance(events, list)
    for s, e, label in events:
        assert 0.0 <= s < e <= 6.0 + 1e-6
        assert label in solution.CLASSES


def test_risk_is_a_probability_for_every_frame(solution, clip):
    cap = cv2.VideoCapture(str(clip))
    est = solution.RiskEstimator()
    est.reset({"video_id": clip.name, "fps": 25.0, "width": 1920, "height": 1080, "n_frames": 150})
    scores = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        scores.append(est.step(frame, i / 25.0))
        i += 1
    assert len(scores) == 150
    assert all(0.0 <= float(s) <= 1.0 for s in scores)
    assert np.isfinite(scores).all()
