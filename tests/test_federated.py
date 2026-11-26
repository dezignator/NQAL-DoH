#!/usr/bin/env python3
"""tests/test_federated.py — validates Eq. 7 and non-IID partition."""
import math
import numpy as np
import torch
import torch.nn as nn
import pytest

from models.federated import SwarmFederatedLearning, FederatedConfig
from models.snn import SNN


@pytest.fixture
def fl():
    model = SNN(input_size=4, hidden_size=16, num_classes=5, use_dsgat=False)
    cfg = FederatedConfig(
        num_clients=3,
        communication_rounds=1,
        local_epochs=1,
        batch_size=8,
        top_k=0.5,
        dp_enabled=False,   # disable DP for deterministic test
        seed=0,
    )
    return SwarmFederatedLearning(model, cfg)


def test_pheromone_weight(fl):
    # sin(0) = 0, sin(small) ≈ small, sin(π/2) = 1
    assert fl._pheromone_weight(0.0) == 0.0
    assert 0.0 < fl._pheromone_weight(0.5) < 1.0
    assert fl._pheromone_weight(math.pi / 2) == pytest.approx(1.0, abs=1e-5)


def test_partition_dirichlet(fl):
    labels = np.random.randint(0, 5, size=200)
    parts = fl.partition_clients(labels)
    assert len(parts) == fl.cfg.num_clients
    total = sum(len(p) for p in parts)
    assert total == len(labels)


def test_client_train_returns_valid_update(fl):
    X = torch.randn(32, 50, 4)
    y = torch.randint(0, 5, (32,))
    criterion = nn.CrossEntropyLoss()
    upd = fl.client_train(0, X, y, criterion)
    assert upd.num_samples == 32
    assert upd.pheromone_weight >= 0.0
    assert upd.bytes_sent > 0


def test_aggregate_pheromone_weighting(fl):
    # Two synthetic updates: one with grad_norm=0 (weight 0), one with grad_norm>0
    from models.federated import ClientUpdate
    w1 = {k: torch.zeros_like(v) for k, v in fl.model.state_dict().items()}
    w2 = {k: torch.ones_like(v) for k, v in fl.model.state_dict().items()}
    upd_zero = ClientUpdate(w1, gradient_norm_sq=0.0, pheromone_weight=0.0,
                            num_samples=10, bytes_sent=100)
    upd_one = ClientUpdate(w2, gradient_norm_sq=1.0, pheromone_weight=1.0,
                           num_samples=10, bytes_sent=100)
    out = fl.aggregate([upd_zero, upd_one])
    # With pheromone 0 on w1, only w2 contributes → tanh(1) ≈ 0.7616
    for k in out:
        # out = tanh(1) + β·QNI_noise(0) ≈ tanh(1)
        expected = math.tanh(1.0)
        assert torch.allclose(out[k], torch.full_like(out[k], expected), atol=1e-3)


def test_top_k_compress(fl):
    grad = torch.randn(100)
    compressed = fl._top_k_compress(grad)
    nonzero = (compressed != 0).sum().item()
    assert nonzero == 50  # top_k=0.5


def test_full_round_runs(fl):
    X = torch.randn(64, 50, 4)
    y = torch.randint(0, 5, (64,))
    clients = [(X[:32], y[:32]), (X[32:], y[32:])]
    diag = fl.step(clients, nn.CrossEntropyLoss(), round_idx=0)
    assert diag["num_updates"] > 0
    assert diag["total_bytes_sent"] > 0
