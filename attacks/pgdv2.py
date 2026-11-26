#!/usr/bin/env python3
"""
attacks/pgd.py
==============
Projected Gradient Descent (PGD) attack — the strong, iterative
adversary the paper benchmarks against (§ 4.2.1).

Default: 10 steps, α = 0.01, ε = 0.05 — the paper's parameters.
"""

from __future__ import annotations

import random
from typing import Optional

import torch
import torch.nn as nn


@torch.enable_grad()
def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.05,
    alpha: float = 0.01,
    steps: int = 10,
    loss_fn: Optional[nn.Module] = None,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    random_start: bool = True,
) -> torch.Tensor:
    """Return the PGD-adversarial example."""
    model.eval()
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    if random_start:
        delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
        x_adv = (x + delta).clamp(clip_min, clip_max).detach()
    else:
        x_adv = x.clone().detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(x_adv)
        loss = loss_fn(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        delta = (x_adv - x).clamp(-epsilon, epsilon)
        x_adv = (x + delta).clamp(clip_min, clip_max).detach()
    return x_adv
