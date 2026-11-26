#!/usr/bin/env python3
"""tests/test_gan.py — validates Eq. 8 and sample shape/ranges."""
import torch
import pytest

from models.gan import ThreatImmunizationGAN, Generator, Discriminator


@pytest.fixture
def gan():
    return ThreatImmunizationGAN(latent_dim=16, seq_len=50, num_features=4)


def test_generator_shape(gan):
    z = torch.randn(8, 16)
    x = gan.G(z)
    assert x.shape == (8, 50, 4)


def test_generator_feature_bounds(gan):
    z = torch.randn(64, 16)
    x = gan.G(z)
    assert (x[..., 0] >= 0).all() and (x[..., 0] <= 16_384).all()
    assert (x[..., 1] >= 0).all() and (x[..., 1] <= 10_000).all()
    assert (x[..., 2] >= 0).all() and (x[..., 2] <= 8).all()
    assert (x[..., 3] >= 0).all() and (x[..., 3] <= 1).all()


def test_discriminator_shape(gan):
    x = torch.randn(8, 50, 4)
    p = gan.D(x)
    assert p.shape == (8, 1)
    assert ((p >= 0) & (p <= 1)).all()


def test_discriminator_features(gan):
    x = torch.randn(8, 50, 4)
    p, f = gan.D(x, return_features=True)
    assert f.shape == (8, 64)


def test_train_step(gan):
    real = torch.randn(32, 50, 4) * 100
    stats = gan.train_step(real)
    assert "d_loss" in stats and "g_loss" in stats
    assert "adv" in stats and "fm" in stats


def test_generate_samples(gan):
    s = gan.generate_adversarial_samples(16)
    assert s.shape == (16, 50, 4)


def test_immunize_runs(gan):
    real = torch.randn(64, 50, 4) * 100
    stats = gan.immunize(real, n_steps=5, batch_size=16, log_every=0)
    assert "d_loss" in stats
