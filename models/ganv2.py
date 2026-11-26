#!/usr/bin/env python3
"""
models/gan.py
=============
Threat Immunization Engine using adversarial GANs (§ 3.7, § 3.8).

Implements Eq. (8):

    min_G max_D  E_{x∼p_data}[log D(x)]
               + E_{z∼p_z}[log(1 − D(G(z)))]
               + γ ‖f_j(G(z))‖_F                                  (8)

where:
    * G : latent → DoH traffic tensor  (T, F)
    * D : traffic → "real" probability
    * f_j : features at discriminator layer j (feature matching term)

The generated samples are *synthetic zero-day anomalies* that the SNN
is trained against (immunisation).  This gives NQAL its 96.3 % zero-day
block rate reported in the paper.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Generator
# =============================================================================
class Generator(nn.Module):
    """
    MLP generator producing DoH traffic tensors of shape (B, T, F).
    """

    def __init__(
        self,
        latent_dim: int = 32,
        seq_len: int = 50,
        num_features: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.num_features = num_features
        self.hidden_dim = hidden_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, seq_len * num_features),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        x = h.view(z.shape[0], self.seq_len, self.num_features)
        # Match the DoH feature bounds:
        # 0: tls_record_len ∈ [0, 16384]
        # 1: inter-arrival  ∈ [0, 10000]
        # 2: entropy        ∈ [0, 8]
        # 3: direction      ∈ {0, 1} — use sigmoid gate
        x = torch.stack([
            torch.sigmoid(x[..., 0]) * 16_384.0,
            torch.sigmoid(x[..., 1]) * 10_000.0,
            torch.sigmoid(x[..., 2]) * 8.0,
            torch.sigmoid(x[..., 3]),
        ], dim=-1)
        return x


# =============================================================================
# Discriminator
# =============================================================================
class Discriminator(nn.Module):
    """
    MLP discriminator with an exposed feature layer for Eq. (8)'s
    γ‖f_j(G(z))‖_F term.
    """

    def __init__(
        self,
        seq_len: int = 50,
        num_features: int = 4,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        in_dim = seq_len * num_features
        self.features = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor:
        h = x.reshape(x.shape[0], -1)
        f = self.features(h)
        p = torch.sigmoid(self.head(f))
        if return_features:
            return p, f
        return p


# =============================================================================
# GAN trainer
# =============================================================================
class ThreatImmunizationGAN:
    """
    Wraps G and D with the Eq. (8) training objective.

    Parameters
    ----------
    latent_dim : int
    seq_len, num_features : int
    gen_hidden, disc_hidden : int
    adv_weight_gamma : float      γ in Eq. (8)
    lr_g, lr_d : float
    """

    def __init__(
        self,
        latent_dim: int = 32,
        seq_len: int = 50,
        num_features: int = 4,
        gen_hidden: int = 64,
        disc_hidden: int = 64,
        adv_weight_gamma: float = 0.10,
        lr_g: float = 2e-4,
        lr_d: float = 2e-4,
        seed: int = 42,
    ) -> None:
        torch.manual_seed(seed)
        self.G = Generator(latent_dim, seq_len, num_features, gen_hidden)
        self.D = Discriminator(seq_len, num_features, disc_hidden)
        self.gamma = float(adv_weight_gamma)
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.num_features = num_features

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.999))
        self._bce = nn.BCELoss()

        self.history: Dict[str, list] = {"g_loss": [], "d_loss": []}

    # ------------------------------------------------------------------
    def _generator_loss(self, real: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Generator loss = adversarial + feature-matching term from Eq. (8).

            L_G = -E[log D(G(z))] + γ‖f_j(G(z)) − f_j(x_real)‖_F
        """
        B = real.shape[0]
        z = torch.randn(B, self.latent_dim, device=real.device)
        fake = self.G(z)

        p_fake, f_fake = self.D(fake, return_features=True)
        _, f_real = self.D(real.detach(), return_features=True)

        adv = -torch.log(p_fake + 1e-8).mean()
        fm = self.gamma * torch.norm(f_fake - f_real.detach(), p="fro") / B
        loss = adv + fm
        return loss, {"adv": float(adv.item()), "fm": float(fm.item())}

    # ------------------------------------------------------------------
    def _discriminator_loss(self, real: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Standard BCE loss on real and fake."""
        B = real.shape[0]
        z = torch.randn(B, self.latent_dim, device=real.device)
        fake = self.G(z).detach()

        p_real = self.D(real)
        p_fake = self.D(fake)

        loss = (
            -torch.log(p_real + 1e-8).mean()
            -torch.log(1.0 - p_fake + 1e-8).mean()
        )
        return loss, {
            "p_real": float(p_real.mean().item()),
            "p_fake": float(p_fake.mean().item()),
        }

    # ------------------------------------------------------------------
    def train_step(self, real: torch.Tensor) -> Dict[str, float]:
        """One Eq. (8) update: D first, then G."""
        # D step
        self.opt_D.zero_grad()
        d_loss, d_stats = self._discriminator_loss(real)
        d_loss.backward()
        self.opt_D.step()

        # G step
        self.opt_G.zero_grad()
        g_loss, g_stats = self._generator_loss(real)
        g_loss.backward()
        self.opt_G.step()

        self.history["d_loss"].append(float(d_loss.item()))
        self.history["g_loss"].append(float(g_loss.item()))
        return {
            "d_loss": float(d_loss.item()),
            "g_loss": float(g_loss.item()),
            **d_stats,
            **g_stats,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_adversarial_samples(
        self, num_samples: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """Sample from G to produce synthetic zero-day anomalies."""
        self.G.eval()
        device = device or next(self.G.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.G(z)

    # ------------------------------------------------------------------
    def immunize(
        self,
        real_data: torch.Tensor,
        n_steps: int = 200,
        batch_size: int = 128,
        log_every: int = 50,
    ) -> Dict[str, float]:
        """Run the GAN training loop over `real_data`."""
        self.G.train()
        self.D.train()
        n = real_data.shape[0]
        last: Dict[str, float] = {}
        for step in range(n_steps):
            idx = torch.randperm(n)[:batch_size]
            batch = real_data[idx]
            last = self.train_step(batch)
            if log_every and (step + 1) % log_every == 0:
                print(
                    f"  [GAN step {step+1:4d}] "
                    f"d_loss={last['d_loss']:.4f}  "
                    f"g_loss={last['g_loss']:.4f}  "
                    f"p_real={last['p_real']:.3f}  "
                    f"p_fake={last['p_fake']:.3f}"
                )
        return last


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 60)
    print(" GAN self-test (Eq. 8)")
    print("=" * 60)

    gan = ThreatImmunizationGAN(latent_dim=16, seq_len=50, num_features=4)
    real = torch.randn(256, 50, 4) * 100  # synthetic data

    # 1) Generator output shape
    z = torch.randn(8, 16)
    fake = gan.G(z)
    assert fake.shape == (8, 50, 4)
    print(f"  G output: {tuple(fake.shape)}")

    # 2) Discriminator output
    p = gan.D(fake)
    assert p.shape == (8, 1)
    assert ((p >= 0) & (p <= 1)).all()
    print(f"  D output range OK")

    # 3) Train step
    for _ in range(10):
        stats = gan.train_step(real[:64])
    print(f"  step stats: d_loss={stats['d_loss']:.3f}, g_loss={stats['g_loss']:.3f}")

    # 4) Generate adversarial samples
    adv = gan.generate_adversarial_samples(32)
    assert adv.shape == (32, 50, 4)
    print(f"  adversarial samples: {tuple(adv.shape)}")

    # 5) Generator produces valid DoH ranges
    assert (adv[..., 0] >= 0).all() and (adv[..., 0] <= 16_384).all()
    assert (adv[..., 1] >= 0).all() and (adv[..., 1] <= 10_000).all()
    assert (adv[..., 2] >= 0).all() and (adv[..., 2] <= 8).all()
    assert (adv[..., 3] >= 0).all() and (adv[..., 3] <= 1).all()
    print(f"  feature bounds OK")

    print("  ✓ GAN self-tests passed.")
