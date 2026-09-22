"""Re-derive phases from the lamp scores stored by signal_timeline.py (after changing thresholds)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from signal_timeline import segments  # noqa: E402

from parivision.signal import fill_phases, phase_from_scores  # noqa: E402

for clip in sys.argv[1:]:
    path = ROOT / "cache/signal" / f"{clip}.json"
    d = json.loads(path.read_text())
    if "scores" not in d:
        print(clip, "has no lamp scores (old format), rerun signal_timeline.py")
        continue
    raw = [phase_from_scores(np.array(s)) for s in d["scores"]]
    d["raw"], d["phases"] = raw, fill_phases(raw, np.array(d["times"]))
    d["segments"] = segments(d["times"], d["phases"])
    path.write_text(json.dumps(d))
    print(clip, " ".join(f"{a:.0f}-{b:.0f}:{p}" for a, b, p in d["segments"]))
