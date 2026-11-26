#!/usr/bin/env python3
"""
models/dsgat.py
===============
Dynamic Spiking Graph Attention (DSGAT).

Implements Eq. (2) of the paper:

    α_ij(t) = softmax( F(s_i(t))ᵀ Θ_Qᵀ Θ_K F(s_j(t)) / √d_k )      (2)

    with F(s) = LIF(∫₀ᵗ e^{-(t−τ)/τ_m} s(τ) dτ)

where:
    * s_i(t) is the spike train of node i at time t
    * F(·)  is the leaky-integral transform (implemented in
            `models.snn.leaky_integral`, also re-exported here)
    * Θ_Q, Θ_K, Θ_V are learned query/key/value projections
    * d_k is the per-head dimension

Graph construction
------------------
For a (B, T, H) spike tensor, we build a graph whose **nodes are
time-steps** and whose **edges** connect timesteps with similar spike
patterns.  This is the natural reading of "dynamic spiking graph
attention" for temporal traffic analysis: the graph structure changes
per input, capturing when in the session the anomaly begins.

Supported graph builders
------------------------
    * knn       : k-nearest-neighbours on spike-train similarity
    * threshold : edges where cosine-sim > θ
    * dense     : fully-connected (baseline)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Leaky integral (duplicated from snn to avoid circular import)
# =============================================================================
def _leaky_integral(spikes: torch.Tensor, tau_m: float = 20.0) -> torch.Tensor:
    """F(s)_t = Σ_{τ=0}^{t} exp(-(t-τ)/τ_m) · s_τ  (O(T) IIR)."""
    decay = math.exp(-1.0 / tau_m)
    B, T, H = spikes.shape
    y = torch.zeros(B, H, device=spikes.device, dtype=spikes.dtype)
    out = []
    for t in range(T):
        y = y * decay + spikes[:, t, :]
        out.append(y)
    return torch.stack(out, dim=1)


# =============================================================================
# Single attention head
# =============================================================================
class DSGATLayer(nn.Module):
    """
    One DSGAT layer = one multi-head attention block over time-steps.

    Shapes:
        in  : (B, T, H_in)
        adj : (B, T, T)  boolean adjacency (1=edge, 0=no edge) or None
        out : (B, T, H_out)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.10,
        tau_m: float = 20.0,
    ) -> None:
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.dropout = dropout
        self.tau_m = tau_m

        # Eq. (2): Θ_Q, Θ_K, Θ_V
        self.theta_q = nn.Linear(in_dim, out_dim, bias=False)
        self.theta_k = nn.Linear(in_dim, out_dim, bias=False)
        self.theta_v = nn.Linear(in_dim, out_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(out_dim, out_dim)

        # Layer norm (helps stabilise training on sparse spikes)
        self.norm = nn.LayerNorm(out_dim)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                       # (B, T, in_dim)
        adj: Optional[torch.Tensor] = None,    # (B, T, T) float {0,1}
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # ---- F(s): leaky integral of the input ----
        f = _leaky_integral(x, self.tau_m)     # (B, T, in_dim)

        # ---- Q, K, V ----
        Q = self.theta_q(f).view(B, T, self.num_heads, self.head_dim)
        K = self.theta_k(f).view(B, T, self.num_heads, self.head_dim)
        V = self.theta_v(f).view(B, T, self.num_heads, self.head_dim)
        # → (B, H, T, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # ---- Attention scores (Eq. 2) ----
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale   # (B, H, T, T)

        # Masking with adjacency
        if adj is not None:
            # adj is (B, T, T); broadcast over heads
            mask = adj.unsqueeze(1)                             # (B, 1, T, T)
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        # Guard against rows that are entirely masked (all -inf)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # ---- Aggregate ----
        h = torch.matmul(attn, V)                               # (B, H, T, d_k)
        h = h.transpose(1, 2).reshape(B, T, self.out_dim)       # (B, T, out_dim)
        h = self.out_proj(h)

        # Residual + LayerNorm
        if self.in_dim == self.out_dim:
            h = h + x
        h = self.norm(h)
        return h


# =============================================================================
# Full DSGAT (2 layers)
# =============================================================================
class DSGAT(nn.Module):
    """
    Two-layer DSGAT with dynamic graph construction.

    Parameters
    ----------
    in_dim, hidden_dim : int
    num_heads : int
    dropout : float
    graph_construction : {"knn", "threshold", "dense"}
    knn_k : int              for "knn"
    edge_threshold : float   for "threshold"
    tau_m : float            leaky-integral time constant
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.10,
        graph_construction: str = "knn",
        knn_k: int = 5,
        edge_threshold: float = 0.10,
        tau_m: float = 20.0,
    ) -> None:
        super().__init__()
        if graph_construction not in ("knn", "threshold", "dense"):
            raise ValueError(f"Unknown graph_construction: {graph_construction}")
        self.graph_construction = graph_construction
        self.knn_k = knn_k
        self.edge_threshold = edge_threshold

        self.layer1 = DSGATLayer(in_dim, hidden_dim, num_heads, dropout, tau_m)
        self.layer2 = DSGATLayer(hidden_dim, hidden_dim, num_heads, dropout, tau_m)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def build_graph(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Build a (B, T, T) adjacency matrix over time-steps.

        Similarity is the cosine similarity of the spike-train vectors
        at each time-step.  This is *dynamic*: a different graph is
        built for every input.
        """
        B, T, H = spikes.shape
        # Normalise rows for cosine similarity
        normed = F.normalize(spikes, dim=-1, eps=1e-6)         # (B, T, H)
        sim = torch.matmul(normed, normed.transpose(-2, -1))   # (B, T, T)

        if self.graph_construction == "dense":
            adj = torch.ones_like(sim)
        elif self.graph_construction == "threshold":
            adj = (sim >= self.edge_threshold).float()
        elif self.graph_construction == "knn":
            k = min(self.knn_k, T - 1)
            # Zero out self-similarity
            sim_masked = sim.clone()
            diag = torch.arange(T, device=sim.device)
            sim_masked[:, diag, diag] = -1.0
            # Top-k neighbours per row
            vals, idx = sim_masked.topk(k, dim=-1)
            adj = torch.zeros_like(sim)
            adj.scatter_(-1, idx, 1.0)
            # Symmetrise: A = max(A, Aᵀ)
            adj = torch.maximum(adj, adj.transpose(-2, -1))
        else:
            raise AssertionError("unreachable")

        # Always include self-loops
        diag_mask = torch.eye(T, device=adj.device).unsqueeze(0)
        adj = torch.clamp(adj + diag_mask, max=1.0)
        return adj

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        adj: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if adj is None:
            adj = self.build_graph(x)
        h = self.layer1(x, adj=adj)
        h = F.relu(h)
        h = self.layer2(h, adj=adj)
        return h


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(" DSGAT self-test (Eq. 2)")
    print("=" * 60)

    B, T, H = 4, 50, 128
    x = (torch.rand(B, T, H, device=device) < 0.3).float()

    for mode in ("knn", "threshold", "dense"):
        dsgat = DSGAT(
            in_dim=H, hidden_dim=64, num_heads=4,
            graph_construction=mode, knn_k=5,
        ).to(device)
        adj = dsgat.build_graph(x)
        assert adj.shape == (B, T, T)
        # Self-loops on diagonal
        eye = torch.eye(T, device=device).unsqueeze(0)
        assert (adj * eye).sum() == B * T, f"{mode}: self-loops missing"
        # Row sums bounded
        assert (adj.sum(dim=-1) >= 1).all(), f"{mode}: dead node"
        out = dsgat(x, adj=adj)
        assert out.shape == (B, T, 64), f"{mode}: wrong output shape"
        print(f"  {mode:10s}: adj_edges={int(adj.sum().item())}, "
              f"out={tuple(out.shape)}")

    # Gradient flow
    x.requires_grad_(True)
    dsgat = DSGAT(in_dim=H, hidden_dim=64).to(device)
    out = dsgat(x)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    print(f"  gradients: |grad| mean = {x.grad.abs().mean().item():.6f}")

    # Multi-head correctness: heads must attend differently
    with torch.no_grad():
        dsgat.eval()
        dsgat.layer1.dropout = 0.0
        f = _leaky_integral(x.detach(), 20.0)
        Q = dsgat.layer1.theta_q(f).view(B, T, 4, 32).transpose(1, 2)
        K = dsgat.layer1.theta_k(f).view(B, T, 4, 32).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(32)
        # Check heads differ
        head_diff = (scores[:, 0] - scores[:, 1]).abs().mean()
        assert head_diff > 1e-4, "Heads produced identical attention"
        print(f"  head divergence: {head_diff.item():.4f}")

    print("  ✓ DSGAT self-tests passed.")
