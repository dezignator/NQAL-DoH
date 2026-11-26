#!/usr/bin/env python3
"""
tests/test_snn.py
=================
End-to-end smoke test for the SNN.
"""
import torch
import torch.nn.functional as F
import pytest

from models.snn import SNN, leaky_integral


@pytest.fixture
def model():
    return SNN(input_size=4, hidden_size=64, num_classes=5, use_dsgat=True)


def test_forward_shape(model):
    x = (torch.rand(8, 50, 4) < 0.3).float()
    logits = model(x)
    assert logits.shape == (8, 5)


def test_features_shape(model):
    x = (torch.rand(8, 50, 4) < 0.3).float()
    feats = model.forward_features(x)
    assert feats.shape == (8, 64)


def test_backward_flows(model):
    x = (torch.rand(4, 50, 4) < 0.3).float()
    y = torch.randint(0, 5, (4,))
    loss = F.cross_entropy(model(x), y)
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    assert n_grad == len(list(model.parameters()))


def test_spike_counts(model):
    x = (torch.rand(2, 50, 4) < 0.3).float()
    counts = model.spike_counts(x)
    assert counts["spikes_total"] > 0
    assert counts["timesteps"] == 50


def test_leaky_integral_saturation():
    x = torch.ones(2, 50, 4)
    y = leaky_integral(x, tau_m=20.0)
    # Constant 1-input → y_t should approach 1/(1-decay) = τ_m
    assert y[0, -1, 0] > 15.0     # asymptote ≈ 20
    assert y[0, -1, 0] < 25.0


def test_no_dsgat_variant():
    model = SNN(input_size=4, hidden_size=64, num_classes=5, use_dsgat=False)
    x = (torch.rand(4, 50, 4) < 0.3).float()
    logits = model(x)
    assert logits.shape == (4, 5)
