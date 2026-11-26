#!/usr/bin/env python3
"""tests/test_pipeline.py — end-to-end integration."""
import torch
import pytest

from models.nqal_pipeline import NQALPipeline, NQALConfig


@pytest.fixture
def pipe():
    return NQALPipeline(NQALConfig(
        hidden_size=16, use_dsgat=True, dsgat_hidden=8, dsgat_heads=2,
        adversarial=False, device="cpu",
    ))


def test_forward(pipe):
    x = (torch.rand(4, 50, 4) < 0.3).float()
    logits = pipe.snn(x)
    assert logits.shape == (4, 5)


def test_train_step(pipe):
    x = (torch.rand(4, 50, 4) < 0.3).float()
    y = torch.randint(0, 5, (4,))
    diag = pipe.train_step(x, y)
    assert "ce" in diag


def test_predict(pipe):
    x = (torch.rand(4, 50, 4) < 0.3).float()
    preds = pipe.predict(x)
    assert preds.shape == (4,)


def test_evaluate(pipe):
    x = (torch.rand(16, 50, 4) < 0.3).float()
    y = torch.randint(0, 5, (16,))
    m = pipe.evaluate(x, y, measure_latency=False, measure_energy=False)
    assert "accuracy" in m
    assert "fpr" in m
    assert 0.0 <= m["accuracy"] <= 1.0


def test_state_roundtrip(pipe):
    sd = pipe.state_dict()
    p2 = NQALPipeline(NQALConfig(
        hidden_size=16, use_dsgat=True, dsgat_hidden=8, dsgat_heads=2,
        adversarial=False, device="cpu",
    ))
    p2.load_state_dict(sd)
    assert torch.allclose(
        pipe.snn.classifier.weight, p2.snn.classifier.weight
    )
