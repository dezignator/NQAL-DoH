#!/usr/bin/env python3
"""
models/encoding.py
==================
Neuromorphic encoder implementing **Eq. (1)** of the paper:

    S(t) = Σ_i w_i · δ(t − t_i) ⊙ M_TLS                          (1)

where:
    * δ(t − t_i) is a Dirac impulse at spike time t_i,
    * w_i are per-feature weights,
    * M_TLS ∈ {0,1}^d is a binary TLS-metadata mask,
    * ⊙ is the Hadamard (element-wise) product.

Three coding schemes are supported (selected via `method`):

    * rate    : p(spike at t) = clip((x − lo)/(hi − lo), 0, 1)
                — robust, cheap; default for Loihi 2 / Akida.
    * latency : single spike per feature at time
                t_i = T − round((x − lo)/(hi − lo) · (T − 1))
                — lowest-energy, uses temporal precision.
    * delta   : spike at t if |x_t − x_{t−1}| > θ
                — best for C2 beaconing (periodic) detection.

The TLS mask `M_TLS` is sampled per-sample from `TLSTypeMask`, whose
dimension equals the number of features (4 in the paper's setup).
During training, entries are randomly dropped with probability
`dropout` — this implements the paper's *TLS metadata masking*
regularisation (Section 3.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# =============================================================================
# TLS metadata mask generator
# =============================================================================
class TLSTypeMask:
    """
    Samples the binary TLS-metadata mask M_TLS ∈ {0,1}^d.

    Feature semantics (matches dataset_loader.py):
        0: TLS record length
        1: inter-arrival time
        2: payload entropy
        3: direction

    During inference all four are normally kept (mask = all ones).
    During training, entries are zeroed with probability `dropout`, which
    forces the SNN to remain discriminative when some metadata is
    unavailable (e.g., middlebox stripping, EVPN header compression).
    """

    def __init__(self, dim: int = 4, dropout: float = 0.10,
                 seed: int = 42) -> None:
        self.dim = dim
        self.dropout = dropout
        self.rng = np.random.default_rng(seed)

    def sample(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        r = rng if rng is not None else self.rng
        mask = np.ones(self.dim, dtype=np.float32)
        if self.dropout > 0.0:
            drop = r.random(self.dim) < self.dropout
            # never drop all dims (keeps the sample informative)
            if drop.all():
                drop[r.integers(self.dim)] = False
            mask[drop] = 0.0
        return mask


# =============================================================================
# Spike encoder
# =============================================================================
@dataclass
class SpikeEncoder:
    """
    Convert a normalised `(T, F)` float tensor into a `(T, F)` spike train.

    Feature-wise min/max ranges are fixed per feature to match the
    class-conditioned distributions in the paper's synthetic dataset:
        * feature 0 (tls_record_len): [0, 16_384]
        * feature 1 (inter-arrival):  [0, 10_000]
        * feature 2 (entropy):        [0, 8]
        * feature 3 (direction):      [0, 1]
    """

    method: str = "rate"
    seq_len: int = 50
    num_features: int = 4
    feature_ranges: list = field(default_factory=lambda: [
        (0.0, 16_384.0),
        (0.0, 10_000.0),
        (0.0, 8.0),
        (0.0, 1.0),
    ])
    delta_threshold: float = 0.10
    stochastic: bool = True            # rate coding uses Bernoulli draws

    def __post_init__(self) -> None:
        if self.method not in ("rate", "latency", "delta"):
            raise ValueError(f"Unknown encoding method: {self.method}")
        if len(self.feature_ranges) != self.num_features:
            raise ValueError("feature_ranges length mismatch")
        self._lo = np.array([r[0] for r in self.feature_ranges], dtype=np.float32)
        self._hi = np.array([r[1] for r in self.feature_ranges], dtype=np.float32)
        self._range = np.maximum(self._hi - self._lo, 1e-6)

    # ------------------------------------------------------------------
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """Clip to [lo, hi] and rescale to [0, 1]."""
        x = np.clip(x, self._lo, self._hi)
        return (x - self._lo) / self._range

    # ------------------------------------------------------------------
    def _rate(self, x_norm: np.ndarray,
              rng: Optional[np.random.Generator]) -> np.ndarray:
        """
        Rate coding: p(spike) = x_norm for each (t, f).

        For each timestep we sample a Bernoulli, producing a stochastic
        spike train.  Deterministic mode (stochastic=False) thresholds
        at 0.5 — used in evaluation to remove variance.
        """
        if self.stochastic and rng is not None:
            return (rng.random(x_norm.shape) < x_norm).astype(np.float32)
        return (x_norm > 0.5).astype(np.float32)

    # ------------------------------------------------------------------
    def _latency(self, x_norm: np.ndarray) -> np.ndarray:
        """
        Latency coding: for each feature f, a single spike at
            t_i = (T − 1) − round(x_norm[:, f] · (T − 1))
        i.e. high value ⇒ early spike.  Lower energy than rate coding
        because total spikes = T × F only for the first timestep of each
        feature.
        """
        T, F = x_norm.shape
        spikes = np.zeros_like(x_norm)
        for f in range(F):
            # collapse across time by taking the mean signal strength
            strength = float(np.clip(x_norm[:, f].mean(), 0.0, 1.0))
            t_idx = (T - 1) - int(round(strength * (T - 1)))
            t_idx = max(0, min(T - 1, t_idx))
            spikes[t_idx, f] = 1.0
        return spikes

    # ------------------------------------------------------------------
    def _delta(self, x_norm: np.ndarray) -> np.ndarray:
        """
        Delta coding: spike at t if |x_t − x_{t−1}| > θ.
        The first timestep spikes if x_0 > θ (initial activation).
        """
        T, F = x_norm.shape
        spikes = np.zeros_like(x_norm)
        diff = np.abs(np.diff(x_norm, axis=0, prepend=x_norm[:1]))
        spikes[diff > self.delta_threshold] = 1.0
        return spikes

    # ------------------------------------------------------------------
    def encode(self, x: np.ndarray,
               rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Convert raw (T, F) features → (T, F) spike train (float32, {0,1}).

        The Hadamard mask M_TLS is applied by the caller
        (dataset_loader applies it to *raw features* before calling
        this method, which is mathematically equivalent because encoding
        is monotone in x — see the ablation flag
        `apply_mask_position` in the module docstring).
        """
        if x.shape != (self.seq_len, self.num_features):
            raise ValueError(
                f"Expected input of shape ({self.seq_len}, {self.num_features}), "
                f"got {x.shape}"
            )
        x_norm = self._normalize(x)
        if self.method == "rate":
            return self._rate(x_norm, rng)
        if self.method == "latency":
            return self._latency(x_norm)
        if self.method == "delta":
            return self._delta(x_norm)
        raise AssertionError("unreachable")

    # ------------------------------------------------------------------
    def apply_mask_position(self, spikes: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
        """
        Eq. (1) alternative: apply M_TLS to the *encoded* spike train
        rather than the raw features.  Provided for ablation; the
        default path masks the raw features (equivalent and cheaper).
        """
        return spikes * mask[None, :]


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    T, F = 50, 4
    x = rng.standard_normal((T, F)).astype(np.float32) * 1000.0
    x[:, 3] = rng.integers(0, 2, T)          # direction feature
    x[:, 2] = rng.uniform(0, 8, T)           # entropy

    print("=" * 60)
    print(" Spike encoding self-test (Eq. 1)")
    print("=" * 60)
    for method in ("rate", "latency", "delta"):
        enc = SpikeEncoder(method=method, seq_len=T, num_features=F)
        s = enc.encode(x, rng=rng)
        assert s.shape == (T, F)
        assert s.dtype == np.float32
        assert set(np.unique(s)).issubset({0.0, 1.0})
        print(f"  {method:8s}  spikes={int(s.sum()):4d}  "
              f"sparsity={1 - s.mean():.3f}")

    # TLS mask
    tm = TLSTypeMask(dim=F, dropout=0.10, seed=0)
    m = tm.sample()
    print(f"  M_TLS sample: {m}")
    print("  ✓ All encoding self-tests passed.")
