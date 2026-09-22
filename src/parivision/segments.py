"""Interval helpers: per-actor intervals -> one non-overlapping segment list per class."""
from __future__ import annotations

import numpy as np


def runs(times: np.ndarray, flags: np.ndarray, max_gap: float) -> list[tuple[float, float]]:
    """Maximal runs of True in a sampled boolean signal; gaps up to ``max_gap`` s are bridged."""
    out: list[list[float]] = []
    for t, f in zip(times, flags):
        if not f:
            continue
        if out and t - out[-1][1] <= max_gap + 1e-9:
            out[-1][1] = float(t)
        else:
            out.append([float(t), float(t)])
    return [(a, b) for a, b in out]


def union(intervals: list[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    """Merge overlapping intervals, and ones separated by at most ``gap`` seconds."""
    merged: list[list[float]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1] + gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(a, b) for a, b in merged]


def finalize(per_class: dict[str, list[tuple[float, float]]], duration: float,
             gap: dict[str, float], min_len: dict[str, float]) -> list[list]:
    """Union per class, drop short blips, clip to the video, emit ``[start, end, label]``."""
    events = []
    for label, ivs in per_class.items():
        for s, e in union(ivs, gap.get(label, 0.0)):
            s, e = max(0.0, s), min(duration, e)
            if e - s >= min_len.get(label, 0.0):
                events.append([round(s, 2), round(e, 2), label])
    events.sort()
    return events
