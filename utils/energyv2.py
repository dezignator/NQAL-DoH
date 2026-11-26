#!/usr/bin/env python3
"""
utils/energy.py
===============
Eq. (9) energy accounting: E_inf = Σ n_i E_spike + λ||W||₀.

This module is deliberately independent of `hardware.energy_profiler`
(the profiler hooks into a live forward pass; this module provides
closed-form estimates for offline analysis in the experiments).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


# Paper-default constants
E_SPIKE_PJ = 0.10          # Loihi 2 / Akida per-spike energy
SPARSITY_LAMBDA = 0.01
GPU_ENERGY_MJ = 5.0        # Reference GPU baseline (§ 4.1)


# -----------------------------------------------------------------------------
def eq9_energy_uj(
    spike_counts: int,
    weights_l0: int,
    e_spike_pj: float = E_SPIKE_PJ,
    sparsity_lambda: float = SPARSITY_LAMBDA,
) -> float:
    """Closed-form Eq. (9): returns energy in μJ."""
    pj = spike_counts * e_spike_pj + sparsity_lambda * weights_l0 * 1e-3
    return pj / 1e6


# -----------------------------------------------------------------------------
def count_l0(model: nn.Module) -> int:
    return sum(int((p != 0).sum()) for p in model.parameters())


# -----------------------------------------------------------------------------
def gpu_energy_ratio(energy_uj: float, gpu_energy_mj: float = GPU_ENERGY_MJ) -> float:
    return float(gpu_energy_mj * 1e3 / max(energy_uj, 1e-9))


# -----------------------------------------------------------------------------
def energy_report(
    model: nn.Module,
    spike_counts: int,
) -> Dict[str, float]:
    l0 = count_l0(model)
    e = eq9_energy_uj(spike_counts, l0)
    return {
        "spike_count": int(spike_counts),
        "weights_l0": int(l0),
        "energy_uj": e,
        "gpu_energy_mj": GPU_ENERGY_MJ,
        "energy_reduction_x": gpu_energy_ratio(e),
    }
