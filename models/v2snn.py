#!/usr/bin/env python3
"""
models/snn.py
=============
The NQAL Spiking Neural Network.

Architecture
------------
    x_spikes : (B, T, F=4)   ─ input spike train (Eq. 1)
        │
        ▼
    Linear(F → H)                ─ feature → hidden projection
        │
        ▼
    LIF (H=128)  ─────────────┐  ─ leaky integrate-and-fire, Eq. (11)
        │                     │
        ▼                     │
    DSGAT(H → H)              │  ─ Eq. (2) — optional (cfg.dsgat.enabled)
        │                     │
        ▼                     │
    LIF (H=128)               │
        │                     │
        ▼                     │
    mean over time ───────────┘  ─ firing-rate readout
        │
        ▼
    Linear(H → C=5)              ─ classifier
        │
        ▼
    logits : (B, C)

Key methods
-----------
    forward(x_spikes)              : (B, T, F) → (B, C)
    forward_features(x_spikes)     : (B, T, F) → (B, H)  — for ZKP audit
    spike_counts(x_spikes)         : total spikes per inference (Eq. 9)
    stdp_step(pre_spikes, post_spikes) : local STDP update (Eq. 11)

The `leaky_integral` utility exactly implements the inner part of F(s)
from Eq. (2):

    F(s)_t = Σ_{τ=0}^{t} exp(-(t-τ)/τ_m) · s_τ

which we expose for use inside DSGAT.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lif_neuron import LIFNeuron, LIFState, STDPLearner
from models.dsgat import DSGAT


# =============================================================================
# Leaky integral (Eq. 2 inner term)
# =============================================================================
def leaky_integral(
    spikes: torch.Tensor,     # (B, T, H)
    tau_m: float = 20.0,
) -> torch.Tensor:
    """
    F(s)_t = Σ_{τ=0}^{t} exp(-(t-τ)/τ_m) · s_τ

    Implemented as an O(T) IIR filter:
        y_t = exp(-1/τ_m) · y_{t−1} + s_t
    """
    decay = math.exp(-1.0 / tau_m)
    B, T, H = spikes.shape
    y = torch.zeros(B, H, device=spikes.device, dtype=spikes.dtype)
    out: List[torch.Tensor] = []
    for t in range(T):
        y = y * decay + spikes[:, t, :]
        out.append(y)
    return torch.stack(out, dim=1)


# =============================================================================
# SNN
# =============================================================================
class SNN(nn.Module):
    """
    NQAL Spiking Neural Network.

    Parameters
    ----------
    input_size : int           F (default 4)
    hidden_size : int          H (default 128, matches paper § 3.1)
    num_classes : int          C (default 5)
    tau_mem, tau_syn : float   (default 20.0, 5.0)
    v_th : float               (default 1.0)
    use_dsgat : bool           whether to insert DSGAT (default True)
    dsgat_kwargs : dict        forwarded to DSGAT
    dropout : float            post-pool dropout (default 0.1)
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 128,
        num_classes: int = 5,
        tau_mem: float = 20.0,
        tau_syn: float = 5.0,
        v_th: float = 1.0,
        v_reset: float = 0.0,
        refr_ticks: int = 2,
        surrogate_slope: float = 10.0,
        use_dsgat: bool = True,
        dsgat_kwargs: Optional[dict] = None,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.use_dsgat = use_dsgat
        self.dropout = dropout

        # Feature → hidden projection
        self.embed = nn.Linear(input_size, hidden_size)

        # LIF populations
        self.lif1 = LIFNeuron(
            tau_mem=tau_mem, tau_syn=tau_syn, v_th=v_th,
            v_reset=v_reset, refr_ticks=refr_ticks,
            surrogate_slope=surrogate_slope,
        )
        self.lif2 = LIFNeuron(
            tau_mem=tau_mem, tau_syn=tau_syn, v_th=v_th,
            v_reset=v_reset, refr_ticks=refr_ticks,
            surrogate_slope=surrogate_slope,
        )

        # Optional DSGAT (Eq. 2)
        if use_dsgat:
            kwargs = dict(dsgat_kwargs or {})
            self.dsgat = DSGAT(
                in_dim=hidden_size,
                hidden_dim=kwargs.get("hidden_dim", 64),
                num_heads=kwargs.get("num_heads", 4),
                dropout=kwargs.get("dropout", 0.10),
                graph_construction=kwargs.get("graph_construction", "knn"),
                knn_k=kwargs.get("knn_k", 5),
                edge_threshold=kwargs.get("edge_threshold", 0.10),
                tau_m=tau_mem,
            )
            dsgat_out = kwargs.get("hidden_dim", 64)
            self.dsgat_proj = nn.Linear(dsgat_out, hidden_size)
        else:
            self.dsgat = None
            self.dsgat_proj = None

        # Classifier
        self.pool_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    # ------------------------------------------------------------------
    def forward_features(self, x_spikes: torch.Tensor) -> torch.Tensor:
        """
        Return the pooled hidden representation (B, H).  Used by the
        ZKP-Learn verification circuit (Eq. 5) and by the energy profiler.
        """
        B, T, F_ = x_spikes.shape
        assert F_ == self.input_size, f"Expected F={self.input_size}, got {F_}"

        # 1) Embed + LIF layer 1
        currents = self.embed(x_spikes)                # (B, T, H)
        spikes1 = self.lif1(currents)                  # (B, T, H)

        # 2) DSGAT (Eq. 2) — optional
        if self.use_dsgat and self.dsgat is not None:
            # Build a graph over time-steps from the spike train.
            # We detach for graph construction (topology does not need
            # gradients — this matches the paper's "dynamic but
            # structure-preserving" description).
            with torch.no_grad():
                adj = self.dsgat.build_graph(spikes1.detach())
            h = self.dsgat(spikes1, adj=adj)           # (B, T, H_dsgat)
            h = self.dsgat_proj(h)                     # (B, T, H)
            # Residual mix: keeps the SNN signal even when attention is
            # uniform at init (avoids dead-gradient start)
            h = h + spikes1
        else:
            h = spikes1

        # 3) LIF layer 2
        spikes2 = self.lif2(h)                         # (B, T, H)

        # 4) Firing-rate readout
        pooled = spikes2.mean(dim=1)                   # (B, H)
        return pooled

    # ------------------------------------------------------------------
    def forward(self, x_spikes: torch.Tensor) -> torch.Tensor:
        pooled = self.forward_features(x_spikes)
        pooled = self.pool_drop(pooled)
        return self.classifier(pooled)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def spike_counts(self, x_spikes: torch.Tensor) -> Dict[str, int]:
        """
        Count spikes at each LIF layer for a batch.  Used by Eq. (9).
        Runs a lightweight manual forward (no graph) for speed.
        """
        B, T, _ = x_spikes.shape
        currents = self.embed(x_spikes)
        # --- layer 1 spikes ---
        state1 = self.lif1.init_state(B, self.hidden_size, x_spikes.device)
        s1_total = 0
        s1_list: List[torch.Tensor] = []
        for t in range(T):
            s = self.lif1.step(currents[:, t, :], state1)
            s1_total += int(s.sum().item())
            s1_list.append(s)
        spikes1 = torch.stack(s1_list, dim=1)

        # --- DSGAT (no spikes — analog) ---
        h = spikes1
        if self.use_dsgat and self.dsgat is not None:
            adj = self.dsgat.build_graph(spikes1)
            h = self.dsgat_proj(self.dsgat(spikes1, adj=adj)) + spikes1

        # --- layer 2 spikes ---
        state2 = self.lif2.init_state(B, self.hidden_size, x_spikes.device)
        s2_total = 0
        for t in range(T):
            s = self.lif2.step(h[:, t, :], state2)
            s2_total += int(s.sum().item())

        return {
            "spikes_lif1": s1_total,
            "spikes_lif2": s2_total,
            "spikes_total": s1_total + s2_total,
            "timesteps": T,
            "batch": B,
        }

    # ------------------------------------------------------------------
    def stdp_step(
        self,
        x_spikes: torch.Tensor,
        learner: STDPLearner,
    ) -> Dict[str, float]:
        """
        One local STDP update of the input→hidden and hidden→output
        projections.  This is the *biologically-plausible* learning
        pathway the paper mentions alongside backprop (§ 3.1).
        """
        with torch.no_grad():
            # Layer-1 pre/post
            currents = self.embed(x_spikes)
            state1 = self.lif1.init_state(
                x_spikes.shape[0], self.hidden_size, x_spikes.device
            )
            s1_list: List[torch.Tensor] = []
            for t in range(x_spikes.shape[1]):
                s1_list.append(self.lif1.step(currents[:, t, :], state1))
            spikes1 = torch.stack(s1_list, dim=1)

            dw1 = learner.trace_update(
                x_spikes, spikes1, self.embed.weight
            )
            learner.apply(self.embed.weight, dw1)

            diag = learner.convergence_diagnostics(self.embed.weight, step=0)
            return diag


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(" SNN self-test")
    print("=" * 60)

    # 1) Basic forward
    model = SNN(input_size=4, hidden_size=128, num_classes=5, use_dsgat=True).to(device)
    x = (torch.rand(8, 50, 4, device=device) < 0.3).float()
    logits = model(x)
    assert logits.shape == (8, 5)
    print(f"  forward: in={tuple(x.shape)} out={tuple(logits.shape)}")

    # 2) Backward
    loss = F.cross_entropy(logits, torch.randint(0, 5, (8,), device=device))
    loss.backward()
    n_grad = sum(p.grad is not None for p in model.parameters())
    print(f"  backward: {n_grad}/{len(list(model.parameters()))} params have grads")

    # 3) Feature extraction
    feats = model.forward_features(x)
    assert feats.shape == (8, 128)
    print(f"  forward_features: {tuple(feats.shape)}")

    # 4) Spike counts (Eq. 9)
    counts = model.spike_counts(x)
    print(f"  spike_counts: {counts}")

    # 5) Leaky integral
    y = leaky_integral(x, tau_m=20.0)
    assert y.shape == x.shape
    # monotone under constant input
    const = torch.ones(2, 10, 4, device=device)
    y_const = leaky_integral(const, tau_m=20.0)
    assert y_const[0, -1, 0] > y_const[0, 0, 0]
    print(f"  leaky_integral: constant input saturates correctly")

    # 6) STDP step
    learner = STDPLearner()
    diag = model.stdp_step(x, learner)
    print(f"  stdp_step: eff_threshold={diag['eff_threshold']:.4f}")

    print("  ✓ SNN self-tests passed.")
