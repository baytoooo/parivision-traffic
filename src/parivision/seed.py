"""Determinism: fixed seeds and deterministic kernels where the backend allows it."""
from __future__ import annotations

import os
import random

import numpy as np

SEED = 2026


def fix_seeds(seed: int = SEED) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
