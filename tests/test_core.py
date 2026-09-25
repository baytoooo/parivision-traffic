"""Fast checks for the pieces that everything else relies on. Run: pytest -q"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parivision import scene as S  # noqa: E402
from parivision.registration import align, warp_points  # noqa: E402
from parivision.segments import finalize, runs, union  # noqa: E402
from parivision.signal import fill_phases, phase_from_scores  # noqa: E402


def test_union_merges_overlaps_and_small_gaps():
    assert union([(0, 2), (1, 3), (5, 6)]) == [(0, 3), (5, 6)]
    assert union([(0, 2), (2.4, 3)], gap=0.5) == [(0, 3)]
    assert union([]) == []


def test_runs_bridges_gaps():
    t = np.arange(0, 3, 0.1)
    flags = (t < 1) | ((t > 1.25) & (t < 2))
    # samples run to 0.9 s, resume at 1.3 s: a 0.4 s hole
    assert runs(t, flags, max_gap=0.45) == [(0.0, t[flags][-1])]
    assert len(runs(t, flags, max_gap=0.3)) == 2


def test_finalize_output_is_valid_for_evaluate():
    events = finalize({"jaywalking": [(1.0, 3.0), (2.5, 4.0), (10.0, 10.3)], "red_light": [(-1.0, 2.0)]},
                      duration=5.0, gap={}, min_len={"jaywalking": 0.5})
    assert events == [[0.0, 2.0, "red_light"], [1.0, 4.0, "jaywalking"]]
    for s, e, _ in events:
        assert 0.0 <= s < e <= 5.0


def test_signal_phase_from_lamp_contrast():
    assert phase_from_scores(np.array([55.0, 1.0, -2.0])) == "red"
    assert phase_from_scores(np.array([0.0, 1.0, 11.0])) == "green"   # green LED is dim in sunlight
    assert phase_from_scores(np.array([-1.0, 46.0, -2.0])) == "yellow"
    assert phase_from_scores(np.array([0.0, 1.0, 2.0])) == "off"


def test_flashing_green_stays_green():
    t = np.arange(0, 4, 0.2)
    raw = ["green", "off"] * 10
    assert set(fill_phases(raw, t)) == {"green"}


def test_scene_masks_cover_the_crossings():
    m = S.masks()
    for polygon in S.CROSSWALKS.values():
        cx, cy = np.asarray(polygon).mean(axis=0).astype(int)
        assert m["crosswalk"][cy, cx] == 1
    assert m["road"].shape == (S.REF_SIZE[1], S.REF_SIZE[0])


def test_registration_recovers_a_shift():
    ref = (np.random.default_rng(0).random((540, 960, 3)) * 255).astype(np.uint8)
    import cv2

    ref = cv2.GaussianBlur(ref, (0, 0), 2)
    shifted = np.roll(ref, (12, -20), axis=(0, 1))
    a = align(shifted, ref, min_inliers=20)
    assert a.ok
    p = warp_points(np.array([[480.0, 270.0]]), a.H)[0]
    assert abs(p[0] - 500) < 2 and abs(p[1] - 258) < 2


def test_metres_per_px_shrinks_towards_the_camera():
    assert S.metres_per_px(700, 200) > S.metres_per_px(700, 900)


def _crash_scene():
    """Two cars meet at t = 5 s (one going east at 6 m/s, one north at 4 m/s, 0.05 m/px) and stand."""
    from parivision.trajectories import Trajectory

    t = np.arange(101) * 0.1
    moving = t <= 5.0
    tt = np.minimum(t, 5.0)
    def car(tid, foot, vel):
        return Trajectory(tid, "vehicle", 2, t, np.zeros((len(t), 4), np.float32), foot, vel,
                          np.full(len(t), 40.0), np.full(len(t), 0.9, np.float32))
    a = car(1000001, np.stack([100 + 120 * tt, np.full_like(t, 500.0)], 1),
            np.stack([np.where(moving, 120.0, 0.0), np.zeros_like(t)], 1))
    b = car(1000002, np.stack([np.full_like(t, 740.0), 900 - 80 * tt], 1),
            np.stack([np.zeros_like(t), np.where(moving, -80.0, 0.0)], 1))
    return [a, b]


def test_collision_rule_finds_a_crash_and_ignores_a_queue():
    from parivision import rules

    ev = rules.collisions(_crash_scene(), lambda x, y: 0.05)
    assert [(round(e.start, 6), round(e.end, 6), e.actors, e.note) for e in ev] == \
        [(4.8, 6.1, [1000001, 1000002], "met at 7 m/s")]
    # the same approach, but the second car stands still throughout (a car joining a queue):
    # its velocity does not change at the contact, so there is no impact
    a, b = _crash_scene()
    b.foot[:] = b.foot[-1]
    b.vel[:] = 0.0
    assert rules.collisions([a, b], lambda x, y: 0.05) == []
