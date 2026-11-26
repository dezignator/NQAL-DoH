#!/usr/bin/env python3
"""
attacks/fgsm.py
===============
Fast Gradient Sign Method (FGSM) attack against a spiking model.

The attack is applied to the *input spike trains* x_spikes ∈ {0,1}^{T×F}
used by NQAL.  Because SNNs are not differentiable through the Heaviside
spike, we route the gradient through the surrogate (already handled by
models/lif_neuron.SurrogateSpike).

Used by `experiments/robustness_study.py` and the paper's § 4.2.1.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.enable_grad()
def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.03,
    loss_fn: Optional[nn.Module] = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """Return x_adv = clamp(x + ε·sign(∇_x L(model(x), y)))."""
    model.eval()
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)
    loss = loss_fn(logits, y)
    grad = torch.autograd.grad(loss, x_adv)[0]
    x_adv = (x_adv + epsilon * grad.sign()).clamp(clip_min, clip_max).detach()
    return x_adv
