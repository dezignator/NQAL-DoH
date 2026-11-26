#!/usr/bin/env python3
"""
utils/reproducibility.py
========================
Deterministic seeding across Python, NumPy, PyTorch, and CUDA.

Usage
-----
    from utils.reproducibility import set_global_seed, make_generator
    set_global_seed(42)
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG we touch in the pipeline."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def make_generator(seed: int, device: str = "cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


@contextmanager
def temp_seed(seed: int):
    """Temporarily seed and restore previous RNG states."""
    states = {
        "py": random.getstate(),
        "np": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    try:
        set_global_seed(seed, deterministic=False)
        yield
    finally:
        random.setstate(states["py"])
        np.random.set_state(states["np"])
        torch.set_rng_state(states["torch"])
        if states["cuda"] is not None:
            torch.cuda.set_rng_state_all(states["cuda"])
