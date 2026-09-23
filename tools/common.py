"""Paths shared by the dev tools."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The sample clips (C3896.MP4 and so on), in samples/ as the README says. PARIVISION_SAMPLES points
# the tools somewhere else; a relative path is taken from the repository root.
SAMPLES = ROOT / os.environ.get("PARIVISION_SAMPLES", "samples")
