#!/usr/bin/env python3
"""
attacks/quantum_noise.py
========================
Quantum Noise Injection (QNI) *attack* — the novel quantum-style
adversary from Table 2 ("QNI Exploits").

This uses a real QuantumRandomWalk sample (from models/qni.py), so the
attack's noise distribution matches the paper's threat model exactly.
Existing systems have **no** defense against this attack (Table 2);
NQAL defeats it via ZKP-validated spike thresholds.
"""

from __future__ import annotations

from typing import Optional

import torch

from models.qni import QuantumRandomWalk


def quantum_noise_attack(
    x: torch.Tensor,
    epsilon: float = 0.05,
    qrw_steps: int = 10,
    n_sites: int = 128,
    seed: int = 42,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """x_adv = clamp(x + ε·QRW_noise, 0, 1)."""
    qrw = QuantumRandomWalk(n_sites=n_sites, steps=qrw_steps, seed=seed)
    noise_np = qrw.sample(x.numel())
    scale = 1.0 / max(qrw.std, 1e-9)
    noise = torch.from_numpy(noise_np).to(x.device).view_as(x)
    noise = noise * scale * epsilon
    return (x + noise).clamp(clip_min, clip_max).detach()
