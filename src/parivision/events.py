"""Rules -> final event list for one clip."""
from __future__ import annotations

import numpy as np

from . import rules as R
from .segments import finalize

# Classes we emit. A class we predict that never occurs in the test set adds a
# zero to the macro average, so classes stay off until they are reliable.
# U-turns round the median nose are detected (the website shows them) but not
# submitted: nothing in view says they are prohibited here, so we cannot tell
# whether the organisers' annotators count them as illegal_u_turn.
ENABLED = ("jaywalking", "failure_to_yield", "red_light", "stop_line", "stopped_vehicle", "wrong_way", "congestion")
SHOWN = ENABLED + ("illegal_u_turn",)

# Segments of one class closer than GAP seconds are joined. The task merges only overlapping
# events, so a car crossing a zebra right after another is a separate event (GAP 0). A jaywalker's
# track breaks whenever a bus passes in front of them, so jaywalking bridges 3 s; congestion and
# stopped vehicles bridge short detection gaps inside one episode.
GAP = {"jaywalking": 3.0, "failure_to_yield": 0.0, "red_light": 0.0, "stop_line": 0.5,
       "stopped_vehicle": 1.0, "wrong_way": 1.0, "congestion": 8.0, "illegal_u_turn": 0.0}
MIN_LEN = {"jaywalking": 1.0, "failure_to_yield": 0.3, "red_light": 0.5, "stop_line": 1.0,
           "stopped_vehicle": 10.0, "wrong_way": 1.5, "congestion": 6.0, "illegal_u_turn": 2.0}


def detect_from_context(ctx: R.Context, H_work_to_ref: np.ndarray) -> tuple[list[list], list[R.Evidence]]:
    evidence: list[R.Evidence] = []
    evidence += R.jaywalking(ctx)
    evidence += R.failure_to_yield(ctx, H_work_to_ref)
    evidence += R.red_light(ctx, H_work_to_ref)
    evidence += R.stop_line(ctx, H_work_to_ref)
    evidence += R.congestion(ctx)
    evidence += R.stopped_vehicle(ctx)
    evidence += R.wrong_way(ctx)
    evidence += R.u_turns(ctx)
    evidence = [ev for ev in evidence if ev.label in SHOWN]
    per_class: dict[str, list[tuple[float, float]]] = {}
    for ev in evidence:
        if ev.label in ENABLED:
            per_class.setdefault(ev.label, []).append((ev.start, ev.end))
    return finalize(per_class, ctx.duration, GAP, MIN_LEN), evidence
