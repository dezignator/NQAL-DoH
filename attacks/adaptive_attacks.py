#!/usr/bin/env python3
"""
attacks/adaptive_attacks.py
===========================
Stronger, adaptive adversaries used for stress-testing (not in the
paper's headline results but required for a rigorous robustness audit).

Includes:
  * **MI-FGSM** (Dong et al., 2018) — momentum
  * **CW-L2-lite** — minimises logit margin
  * **EOT over quantum noise** — averages gradient across QRW draws
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.qni import QuantumRandomWalk


@torch.enable_grad()
def mi_fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.05,
    steps: int = 10,
    decay: float = 1.0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """Momentum Iterative FGSM."""
    model.eval()
    x_adv = x.clone().detach()
    g = torch.zeros_like(x)
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        grad = grad / (grad.abs().mean() + 1e-9)
        g = decay * g + grad
        x_adv = x_adv.detach() + (epsilon / steps) * g.sign()
        delta = (x_adv - x).clamp(-epsilon, epsilon)
        x_adv = (x + delta).clamp(clip_min, clip_max).detach()
    return x_adv


@torch.enable_grad()
def cw_l2_lite(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.05,
    steps: int = 20,
    lr: float = 0.01,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """CW-style logit-margin minimisation under L∞ budget ε."""
    model.eval()
    delta = torch.zeros_like(x, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    onehot = F.one_hot(y, num_classes=model.num_classes).float()
    for _ in range(steps):
        opt.zero_grad()
        logits = model((x + delta).clamp(clip_min, clip_max))
        correct = (logits * onehot).sum(-1, keepdim=True)
        other = (logits - onehot * 1e9).max(-1, keepdim=True).values
        loss = (other - correct).clamp(min=0).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():
            delta.data.clamp_(-epsilon, epsilon)
    return (x + delta.detach()).clamp(clip_min, clip_max)


@torch.enable_grad()
def eot_quantum_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float = 0.05,
    n_samples: int = 5,
    qrw_steps: int = 10,
    n_sites: int = 128,
    seed: int = 0,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> torch.Tensor:
    """
    Expectation-over-Transform quantum attack:
    gradient is averaged across `n_samples` QRW draws to defeat
    stochastic defenses like QNI.
    """
    model.eval()
    qrw = QuantumRandomWalk(n_sites=n_sites, steps=qrw_steps, seed=seed)
    scale = 1.0 / max(qrw.std, 1e-9)
    x_adv = x.clone().detach().requires_grad_(True)
    total_grad = torch.zeros_like(x_adv)
    for _ in range(n_samples):
        noise = torch.from_numpy(qrw.sample(x.numel())).to(x.device).view_as(x)
        noise = noise * scale * epsilon
        x_noisy = (x_adv + noise).clamp(clip_min, clip_max)
        loss = F.cross_entropy(model(x_noisy), y)
        grad = torch.autograd.grad(loss, x_adv, retain_graph=True)[0]
        total_grad = total_grad + grad
    x_adv = (x_adv + epsilon * (total_grad / n_samples).sign())
    return x_adv.clamp(clip_min, clip_max).detach()
