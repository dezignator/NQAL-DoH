#!/usr/bin/env python3
"""
models/nqal_pipeline.py
=======================
The NQAL pipeline — glue module that combines every component of the
paper's architecture (§ 3, Figure 1):

    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  SNN + DSGAT │  │  QNI + Adv   │  │  ZKP-Learn   │
    │  (Eq. 1, 2,  │  │  loss (Eq. 3,│  │  (Eq. 5, 6)  │
    │   11)        │  │   4, 10)     │  │              │
    └──────────────┘  └──────────────┘  └──────────────┘
             │              │                  │
             └──────────────┴──────────────────┘
                            │
                    ┌───────┴────────┐
                    │  NQALPipeline  │
                    └───────┬────────┘
                            │
              ┌─────────────┴──────────────┐
              │                            │
      ┌───────▼───────┐            ┌───────▼───────┐
      │ Swarm-FL      │            │ GAN           │
      │ (Eq. 7)       │            │ immunization  │
      │               │            │ (Eq. 8)       │
      └───────────────┘            └───────────────┘

The class exposes the exact interface used by `train.py` and
`evaluate.py`:

    predict()                  : inference (with optional ZKP)
    train_step()               : one supervised + adversarial step
    federated_round()          : one Eq. (7) FL round
    immunize()                 : run GAN to precondition against zero-days
    verify_anomaly()           : ZKP audit of a flagged sample
    spike_counts()             : Eq. (9) energy accounting
    evaluate()                 : full metric suite (accuracy, FPR,
                                 robustness, latency, energy)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.snn import SNN
from models.qni import QNI
from models.zkp import ZKPLearn, ZKProof
from models.gan import ThreatImmunizationGAN
from models.federated import SwarmFederatedLearning, FederatedConfig


# =============================================================================
# Config
# =============================================================================
@dataclass
class NQALConfig:
    # Data
    seq_len: int = 50
    num_features: int = 4
    num_classes: int = 5
    # SNN
    hidden_size: int = 128
    tau_mem: float = 20.0
    tau_syn: float = 5.0
    v_th: float = 1.0
    use_dsgat: bool = True
    dsgat_hidden: int = 64
    dsgat_heads: int = 4
    # QNI
    qni_eta: float = 0.05
    qni_steps: int = 10
    qni_lambda_kl: float = 0.10
    qni_epsilon: float = 0.05
    # ZKP
    zkp_tau: float = 0.70
    zkp_use_rate: bool = True
    # GAN
    gan_latent: int = 32
    gan_gamma: float = 0.10
    # FL
    fl_num_clients: int = 10
    fl_rounds: int = 20
    fl_local_epochs: int = 3
    fl_beta: float = 0.10
    fl_top_k: float = 0.10
    # Training
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    adversarial: bool = True
    seed: int = 42
    device: str = "cpu"


# =============================================================================
# Pipeline
# =============================================================================
class NQALPipeline:
    """
    End-to-end NQAL pipeline.

    Parameters
    ----------
    cfg : NQALConfig
    """

    def __init__(self, cfg: NQALConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        torch.manual_seed(cfg.seed)

        # 1) SNN + DSGAT (Eqs. 1, 2, 11)
        self.snn = SNN(
            input_size=cfg.num_features,
            hidden_size=cfg.hidden_size,
            num_classes=cfg.num_classes,
            tau_mem=cfg.tau_mem,
            tau_syn=cfg.tau_syn,
            v_th=cfg.v_th,
            use_dsgat=cfg.use_dsgat,
            dsgat_kwargs={
                "hidden_dim": cfg.dsgat_hidden,
                "num_heads": cfg.dsgat_heads,
            },
        ).to(self.device)

        # 2) QNI (Eqs. 3, 4, 10)
        self.qni = QNI(
            eta=cfg.qni_eta,
            qrw_steps=cfg.qni_steps,
            lambda_kl=cfg.qni_lambda_kl,
            epsilon=cfg.qni_epsilon,
            seed=cfg.seed,
        ).to(self.device)

        # 3) ZKP-Learn (Eqs. 5, 6)
        self.zkp = ZKPLearn(
            threshold_tau=cfg.zkp_tau,
            use_rate=cfg.zkp_use_rate,
            seed=cfg.seed,
        )

        # 4) GAN (Eq. 8)
        self.gan = ThreatImmunizationGAN(
            latent_dim=cfg.gan_latent,
            seq_len=cfg.seq_len,
            num_features=cfg.num_features,
            adv_weight_gamma=cfg.gan_gamma,
            seed=cfg.seed,
        )

        # 5) Swarm-FL (Eq. 7)
        self.fl = SwarmFederatedLearning(
            global_model=self.snn,
            cfg=FederatedConfig(
                num_clients=cfg.fl_num_clients,
                communication_rounds=cfg.fl_rounds,
                local_epochs=cfg.fl_local_epochs,
                swarm_beta=cfg.fl_beta,
                top_k=cfg.fl_top_k,
                batch_size=cfg.batch_size,
                seed=cfg.seed,
            ),
            qni=self.qni,
            device=self.device,
        )

        # Optimizer for centralised training
        self.opt = torch.optim.Adam(
            self.snn.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        # Training state
        self.global_step = 0
        self.epoch = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train_step(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Dict[str, float]:
        """
        One supervised training step.

        If cfg.adversarial: uses Eq. (4) loss  (clean CE + λ·KL on PGD).
        Otherwise:          standard CE only.
        """
        self.snn.train()
        x = x.to(self.device)
        y = y.to(self.device)

        self.opt.zero_grad()

        if self.cfg.adversarial:
            loss, diag = self.qni.adversarial_loss(self.snn, x, y)
        else:
            logits = self.snn(x)
            loss = self.criterion(logits, y)
            diag = {"ce": float(loss.item()), "loss": float(loss.item())}

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.snn.parameters(), 1.0)
        self.opt.step()
        self.global_step += 1
        return diag

    # ------------------------------------------------------------------
    def federated_round(
        self,
        clients_data: List[tuple],
    ) -> Dict[str, float]:
        """One Eq. (7) federated round."""
        return self.fl.step(clients_data, self.criterion, round_idx=self.epoch)

    # ------------------------------------------------------------------
    def immunize(
        self, real_data: torch.Tensor, n_steps: int = 200
    ) -> Dict[str, float]:
        """Train the GAN (Eq. 8) to generate zero-day anomalies."""
        return self.gan.immunize(real_data, n_steps=n_steps)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """Inference.  Returns class predictions (B,) or logits."""
        self.snn.eval()
        logits = self.snn(x.to(self.device))
        return logits if return_logits else logits.argmax(-1)

    # ------------------------------------------------------------------
    def verify_anomaly(
        self,
        x: torch.Tensor,
        sample_id: int,
        public_input: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Run ZKP-Learn on a single sample.  Returns proof + verification
        result + privacy audit.
        """
        # Use the SNN's spike train as the ZKP input
        with torch.no_grad():
            currents = self.snn.embed(x.to(self.device).unsqueeze(0))
            state = self.snn.lif1.init_state(1, self.cfg.hidden_size, self.device)
            spike_list = []
            for t in range(self.cfg.seq_len):
                spike_list.append(self.snn.lif1.step(currents[:, t, :], state))
            spike_train = torch.stack(spike_list, dim=1)[0]  # (T, H)

        try:
            proof = self.zkp.prove(spike_train, sample_id, public_input)
            verified = self.zkp.verify(proof, sample_id, public_input)
            audit = self.zkp.privacy_audit(spike_train, sample_id)
            return {
                "proof_generated": True,
                "verified": verified,
                **audit,
            }
        except ValueError as e:
            return {
                "proof_generated": False,
                "verified": False,
                "reason": str(e),
            }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        adversarial_eps: float = 0.03,
        measure_energy: bool = True,
        measure_latency: bool = True,
    ) -> Dict[str, float]:
        """Full metric suite matching the paper's Figure 1 targets."""
        self.snn.eval()
        X = X.to(self.device)
        y = y.to(self.device)

        # --- Clean accuracy + FPR ---
        with torch.no_grad():
            logits = self.snn(X)
            preds = logits.argmax(-1)
        acc = float((preds == y).float().mean())
        # FPR: benign (class 0) misclassified as any other
        benign = y == 0
        fpr = float(((preds != 0) & benign).float().sum() / max(benign.sum().item(), 1))

        # --- Adversarial robustness (FGSM) ---
        X_adv = X.clone().detach().requires_grad_(True)
        out = self.snn(X_adv)
        loss = F.cross_entropy(out, y)
        grad = torch.autograd.grad(loss, X_adv)[0]
        X_adv = (X_adv + adversarial_eps * grad.sign()).clamp(0, 1).detach()
        with torch.no_grad():
            adv_acc = float((self.snn(X_adv).argmax(-1) == y).float().mean())

        # --- Per-class detection rate (Table 2) ---
        detection = {}
        for c, name in enumerate(
            ["benign", "dns_tunneling", "c2_beaconing", "dga", "adv_doh"]
        ):
            mask = y == c
            if mask.sum() > 0:
                detection[name] = float((preds[mask] == c).float().mean())

        result: Dict[str, float] = {
            "accuracy": acc,
            "fpr": fpr,
            "adversarial_robustness": adv_acc,
            **{f"detection_{k}": v for k, v in detection.items()},
        }

        # --- Latency ---
        if measure_latency:
            with torch.no_grad():
                t0 = time.perf_counter()
                for _ in range(20):
                    _ = self.snn(X[:1])
                t1 = time.perf_counter()
            result["latency_ms"] = (t1 - t0) / 20 * 1000.0

        # --- Energy (Eq. 9) ---
        if measure_energy:
            counts = self.snn.spike_counts(X[:1])
            result["spikes_per_query"] = counts["spikes_total"]
            result["energy_uj"] = self._estimate_energy(counts["spikes_total"])

        return result

    # ------------------------------------------------------------------
    def _estimate_energy(self, n_spikes: int) -> float:
        """Eq. (9): E = n·E_spike + λ·||W||₀, in μJ."""
        e_spike_pj = 0.10
        l0 = sum(float((p != 0).sum()) for p in self.snn.parameters())
        energy_pj = n_spikes * e_spike_pj + 0.01 * l0 * 1e-3
        return energy_pj / 1e6

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, object]:
        return {
            "snn": self.snn.state_dict(),
            "gan_G": self.gan.G.state_dict(),
            "gan_D": self.gan.D.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
        }

    def load_state_dict(self, sd: Dict[str, object]) -> None:
        self.snn.load_state_dict(sd["snn"])
        self.gan.G.load_state_dict(sd["gan_G"])
        self.gan.D.load_state_dict(sd["gan_D"])
        self.global_step = int(sd.get("global_step", 0))
        self.epoch = int(sd.get("epoch", 0))


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 60)
    print(" NQAL Pipeline self-test")
    print("=" * 60)

    cfg = NQALConfig(
        hidden_size=32,
        use_dsgat=True,
        dsgat_hidden=16,
        adversarial=True,
        fl_num_clients=2,
        device="cpu",
    )
    pipe = NQALPipeline(cfg)

    # 1) Forward
    x = (torch.rand(8, 50, 4) < 0.3).float()
    y = torch.randint(0, 5, (8,))
    logits = pipe.snn(x)
    assert logits.shape == (8, 5)
    print(f"  forward: {tuple(x.shape)} → {tuple(logits.shape)}")

    # 2) Train step
    diag = pipe.train_step(x, y)
    print(f"  train_step: ce={diag.get('ce', 0):.4f}, "
          f"kl={diag.get('kl', 0):.4f}")

    # 3) Predict
    preds = pipe.predict(x)
    assert preds.shape == (8,)
    print(f"  predict: {preds.tolist()}")

    # 4) ZKP verify
    audit = pipe.verify_anomaly(x[:1], sample_id=42)
    print(f"  ZKP: verified={audit['verified']}, "
          f"proof={audit.get('proof_size_bytes', 0)} B")

    # 5) Evaluate
    metrics = pipe.evaluate(x, y)
    print(f"  evaluate: acc={metrics['accuracy']:.4f}, "
          f"fpr={metrics['fpr']:.4f}, "
          f"energy={metrics.get('energy_uj', 0):.4f} μJ")

    # 6) State round-trip
    sd = pipe.state_dict()
    pipe2 = NQALPipeline(cfg)
    pipe2.load_state_dict(sd)
    assert torch.allclose(
        pipe.snn.classifier.weight, pipe2.snn.classifier.weight
    )
    print(f"  state round-trip: OK")

    print("  ✓ Pipeline self-tests passed.")
