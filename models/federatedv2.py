#!/usr/bin/env python3
"""
models/federated.py
===================
Swarm-Based Federated Learning (§ 3.5, § 3.6).

Implements Eq. (7) of the paper:

    Δw_i^{k+1} = φ( Σ_{j∈N_k} w_j^k · sin(g_j, g_j)
                    / Σ_{j∈N_k} sin(g_j, g_j) )                  (7)
                + β · QNI(Δw_i^k)

Interpretation (documented, since the paper's notation is compact)
-----------------------------------------------------------------
    g_j       : client j's *local gradient* (or update direction)
    sin(g_j,g_j) → sin(‖g_j‖²)   : pheromone-inspired scalar weight
                                    (bio-inspired: stigmergic signalling
                                    modulated by local gradient energy)
    w_j^k     : client j's weights at round k
    N_k       : neighbourhood of client i (default: all clients)
    φ         : pheromone nonlinearity (tanh / sigmoid / relu)
    β         : QNI noise coefficient
    QNI(·)    : quantum noise injection from `models.qni`

The optional QNI term is what the paper adds to make the update
adversarially robust: each round's aggregate carries a small quantum-
walk perturbation.

Additional features (all paper-aligned, § 3.5)
----------------------------------------------
* Dirichlet non-IID client partition (α controls skew).
* Top-k gradient compression (default 10 %) to reduce comms.
* Gaussian differential privacy (DP-SGD style) applied client-side.
* Communication accounting (bytes exchanged per round) so we can
  reproduce the paper's "reduced communication overhead" claim.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from models.qni import QNI


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class FederatedConfig:
    num_clients: int = 10
    communication_rounds: int = 20
    local_epochs: int = 3
    client_fraction: float = 1.0
    batch_size: int = 64
    learning_rate: float = 1e-3
    swarm_beta: float = 0.10
    pheromone_fn: str = "tanh"            # tanh | sigmoid | relu
    dirichlet_alpha: float = 0.5
    top_k: float = 0.10                   # fraction of coords kept
    dp_enabled: bool = True
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    qni_eta: float = 0.05
    qni_steps: int = 10
    seed: int = 42


# =============================================================================
# Client-side helper
# =============================================================================
@dataclass
class ClientUpdate:
    """What a client sends back to the aggregator."""
    weights: Dict[str, torch.Tensor]
    gradient_norm_sq: float
    pheromone_weight: float
    num_samples: int
    bytes_sent: int


# =============================================================================
# Swarm-FL
# =============================================================================
class SwarmFederatedLearning:
    """
    Federated learning with swarm (pheromone) aggregation (Eq. 7).

    Usage
    -----
        fl = SwarmFederatedLearning(global_model, cfg)
        for round_k in range(cfg.communication_rounds):
            updates = fl.run_round(client_data, criterion)
            fl.aggregate(updates, round_k)
    """

    def __init__(
        self,
        global_model: nn.Module,
        cfg: FederatedConfig,
        qni: Optional[QNI] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = global_model
        self.cfg = cfg
        self.device = device or torch.device("cpu")
        self.qni = qni or QNI(
            eta=cfg.qni_eta, qrw_steps=cfg.qni_steps, seed=cfg.seed
        )
        self.rng = np.random.default_rng(cfg.seed)

        if cfg.pheromone_fn == "tanh":
            self.phi: Callable[[torch.Tensor], torch.Tensor] = torch.tanh
        elif cfg.pheromone_fn == "sigmoid":
            self.phi = torch.sigmoid
        elif cfg.pheromone_fn == "relu":
            self.phi = torch.relu
        else:
            raise ValueError(f"Unknown pheromone_fn: {cfg.pheromone_fn}")

        # DP noise scale: for Gaussian mechanism with (ε, δ)-DP,
        # σ ≥ √(2·ln(1.25/δ)) / ε · Δf.  We use Δf = 1 (gradient clipped to L2=1).
        self._dp_sigma = (
            math.sqrt(2.0 * math.log(1.25 / cfg.dp_delta)) / cfg.dp_epsilon
        )

        # Tracking
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    def partition_clients(
        self, labels: np.ndarray
    ) -> List[np.ndarray]:
        """
        Dirichlet partition of sample indices across clients.
        Non-IID controlled by `cfg.dirichlet_alpha`.
        """
        rng = self.rng
        n_classes = int(labels.max()) + 1
        n = self.cfg.num_clients
        clients: List[List[int]] = [[] for _ in range(n)]
        for c in range(n_classes):
            idx = np.where(labels == c)[0]
            rng.shuffle(idx)
            props = rng.dirichlet([self.cfg.dirichlet_alpha] * n)
            cuts = (np.cumsum(props) * len(idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(idx, cuts)):
                clients[k].extend(chunk.tolist())
        return [np.array(sorted(c), dtype=np.int64) for c in clients]

    # ------------------------------------------------------------------
    def _top_k_compress(self, grad: torch.Tensor) -> torch.Tensor:
        """Keep only top-k magnitude coordinates (sparsify)."""
        if self.cfg.top_k >= 1.0:
            return grad
        k = max(1, int(self.cfg.top_k * grad.numel()))
        flat = grad.view(-1)
        _, idx = flat.abs().topk(k)
        out = torch.zeros_like(flat)
        out[idx] = flat[idx]
        return out.view_as(grad)

    # ------------------------------------------------------------------
    def _dp_noise(self, grad: torch.Tensor) -> torch.Tensor:
        """Add Gaussian DP noise (client-side)."""
        if not self.cfg.dp_enabled:
            return grad
        return grad + torch.randn_like(grad) * self._dp_sigma

    # ------------------------------------------------------------------
    def _pheromone_weight(self, grad_norm_sq: float) -> float:
        """
        Eq. (7): sin(g_i, g_i) ≡ sin(‖g_i‖²).
        We clamp to a small angle to avoid the periodic wrap-around that
        sin() introduces for large norms, matching the paper's
        "pheromone inspiration" (bounded signalling strength).
        """
        return float(math.sin(min(grad_norm_sq, math.pi / 2)))

    # ------------------------------------------------------------------
    def client_train(
        self,
        client_idx: int,
        X: torch.Tensor,
        y: torch.Tensor,
        criterion: nn.Module,
    ) -> ClientUpdate:
        """Train one client locally and return its update."""
        local_model = copy.deepcopy(self.model).to(self.device)
        local_model.load_state_dict(self.model.state_dict())
        local_model.train()
        opt = torch.optim.Adam(
            local_model.parameters(), lr=self.cfg.learning_rate
        )

        X = X.to(self.device)
        y = y.to(self.device)
        n = X.shape[0]

        for epoch in range(self.cfg.local_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.cfg.batch_size):
                idx = perm[i:i + self.cfg.batch_size]
                xb, yb = X[idx], y[idx]
                opt.zero_grad()
                loss = criterion(local_model(xb), yb)
                loss.backward()
                # Gradient clipping (also bounds DP sensitivity)
                torch.nn.utils.clip_grad_norm_(local_model.parameters(), 1.0)
                # DP noise (client-side, on gradients)
                for p in local_model.parameters():
                    if p.grad is not None:
                        p.grad.data = self._dp_noise(p.grad.data)
                opt.step()

        # Collect weights + overall gradient norm (approx via delta from global)
        weights = {k: v.detach().clone() for k, v in local_model.state_dict().items()}
        diff_sq = 0.0
        for k, v in weights.items():
            g = self.model.state_dict()[k]
            diff_sq += float((v - g).pow(2).sum().item())

        # Byte accounting
        bytes_sent = sum(v.numel() * 4 for v in weights.values())
        # After top-k compression:
        if self.cfg.top_k < 1.0:
            for k in weights:
                weights[k] = self._top_k_compress(weights[k])
            bytes_sent = int(bytes_sent * self.cfg.top_k)

        pheromone = self._pheromone_weight(diff_sq)

        return ClientUpdate(
            weights=weights,
            gradient_norm_sq=diff_sq,
            pheromone_weight=pheromone,
            num_samples=n,
            bytes_sent=bytes_sent,
        )

    # ------------------------------------------------------------------
    def run_round(
        self,
        clients_data: List[tuple],
        criterion: nn.Module,
    ) -> List[ClientUpdate]:
        """Run one communication round: all clients train locally."""
        n_clients = len(clients_data)
        selected = self.rng.choice(
            n_clients,
            size=max(1, int(self.cfg.client_fraction * n_clients)),
            replace=False,
        )
        updates: List[ClientUpdate] = []
        for idx in selected:
            X, y = clients_data[idx]
            updates.append(
                self.client_train(int(idx), X, y, criterion)
            )
        return updates

    # ------------------------------------------------------------------
    def aggregate(
        self, updates: List[ClientUpdate]
    ) -> Dict[str, torch.Tensor]:
        """
        Eq. (7): pheromone-weighted aggregation + φ + β·QNI noise.
        Returns the new global weight dict.
        """
        # Weighted average over pheromone
        total_w = sum(u.pheromone_weight for u in updates) + 1e-9
        global_weights: Dict[str, torch.Tensor] = {}

        for k in updates[0].weights:
            acc = None
            for u in updates:
                term = u.weights[k] * u.pheromone_weight
                acc = term if acc is None else acc + term
            avg = acc / total_w
            # Apply φ
            phi_avg = self.phi(avg)
            # β · QNI(Δw) — Δw is implicitly `avg`
            self.qni.train()
            noise = self.qni.inject_weights(avg.detach())
            global_weights[k] = phi_avg + self.cfg.swarm_beta * (noise - avg.detach())
            self.qni.eval()

        # Commit to the model
        self.model.load_state_dict(global_weights, strict=False)
        return global_weights

    # ------------------------------------------------------------------
    def step(
        self,
        clients_data: List[tuple],
        criterion: nn.Module,
        round_idx: int,
    ) -> Dict[str, float]:
        """One complete FL step (train + aggregate + log)."""
        updates = self.run_round(clients_data, criterion)
        self.aggregate(updates)

        bytes_sent = sum(u.bytes_sent for u in updates)
        diag = {
            "round": round_idx,
            "num_updates": len(updates),
            "total_bytes_sent": bytes_sent,
            "mean_pheromone": float(np.mean([u.pheromone_weight for u in updates])),
            "mean_grad_norm_sq": float(np.mean([u.gradient_norm_sq for u in updates])),
        }
        self.history.append(diag)
        return diag


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from models.snn import SNN

    torch.manual_seed(0)
    print("=" * 60)
    print(" Swarm Federated Learning self-test (Eq. 7)")
    print("=" * 60)

    # Model
    model = SNN(input_size=4, hidden_size=32, num_classes=5, use_dsgat=False)
    cfg = FederatedConfig(
        num_clients=4,
        communication_rounds=2,
        local_epochs=1,
        batch_size=16,
        top_k=0.5,
        dp_enabled=True,
        dp_epsilon=2.0,
    )
    fl = SwarmFederatedLearning(model, cfg)

    # Synthetic data
    labels = np.random.randint(0, 5, size=200)
    partition = fl.partition_clients(labels)
    clients_data = [
        (
            torch.from_numpy(np.random.randn(len(partition[i]), 50, 4).astype(np.float32)),
            torch.from_numpy(labels[partition[i]]),
        )
        for i in range(cfg.num_clients)
    ]

    criterion = nn.CrossEntropyLoss()
    for r in range(cfg.communication_rounds):
        diag = fl.step(clients_data, criterion, round_idx=r)
        print(f"  round {r}: updates={diag['num_updates']}, "
              f"bytes={diag['total_bytes_sent']}, "
              f"pheromone_mean={diag['mean_pheromone']:.4f}")

    # Verify weights changed
    w_before = model.classifier.weight.detach().clone()
    for r in range(cfg.communication_rounds):
        fl.step(clients_data, criterion, round_idx=100 + r)
    w_after = model.classifier.weight.detach()
    delta = (w_after - w_before).abs().sum()
    assert delta > 1e-4, "FL did not modify model weights"
    print(f"  |ΔW| after 2 more rounds: {delta:.6f}")
    print("  ✓ Federated self-tests passed.")
