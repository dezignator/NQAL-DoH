#!/usr/bin/env python3
"""
tests/test_qni.py
=================
Validates QNI (Eq. 3), adversarial loss (Eq. 4), and Theorem 1 (Eq. 10).
"""
import torch
import pytest

from models.qni import QNI, QuantumRandomWalk
from models.snn import SNN


def test_qrw_distribution():
    qrw = QuantumRandomWalk(n_sites=128, steps=10)
    d = qrw.distribution
    assert abs(d.sum() - 1.0) < 1e-6
    assert (d >= 0).all()


def test_qrw_ballistic():
    """Quantum walk spreads faster than classical (std ∝ t)."""
    qrw_5 = QuantumRandomWalk(n_sites=256, steps=5)
    qrw_10 = QuantumRandomWalk(n_sites=256, steps=10)
    # Doubling steps should ≈ double std (ballistic), not √2
    ratio = qrw_10.std / qrw_5.std
    assert ratio > 1.5, f"QRW ballistic expected, got ratio {ratio}"


def test_inject_weights():
    qni = QNI(eta=0.05)
    qni.train()
    W = torch.randn(32, 32)
    W_pert = qni.inject_weights(W)
    delta = (W_pert - W).norm() / W.norm()
    assert delta > 0
    assert delta < 0.5


def test_inject_weights_noop_in_eval():
    qni = QNI(eta=0.05, training_only=True)
    qni.eval()
    W = torch.randn(32, 32)
    W_pert = qni.inject_weights(W)
    assert torch.allclose(W, W_pert)


def test_adversarial_loss():
    model = SNN(input_size=4, hidden_size=32, num_classes=5, use_dsgat=False)
    model.train()
    x = (torch.rand(8, 50, 4) < 0.3).float()
    y = torch.randint(0, 5, (8,))
    qni = QNI()
    loss, diag = qni.adversarial_loss(model, x, y)
    assert loss.requires_grad
    assert diag["ce"] > 0
    assert 0.0 <= diag["adv_acc"] <= 1.0


def test_theorem1():
    model = SNN(input_size=4, hidden_size=32, num_classes=5, use_dsgat=False)
    model.eval()
    qni = QNI(eta=0.05)
    qni.eval()
    x = (torch.rand(4, 50, 4) < 0.3).float()
    diag = qni.verify_theorem1(model, x, delta=0.5, num_samples=32)
    assert diag["satisfied"], f"Bound violated: {diag}"
