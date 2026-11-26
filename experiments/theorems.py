#!/usr/bin/env python3
"""
experiments/theorems.py
=======================
Empirical validation of the paper's two theorems:

    Theorem 1 — QNI robustness bound (Eq. 10)
        P(||f(x+q) − f(x)|| ≥ δ) ≤ L_f² σ² / δ²

    Theorem 2 — STDP convergence (Eq. 11)
        Effective threshold converges (std-of-late ≤ std-of-early)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from models.nqal_pipeline import NQALPipeline, NQALConfig
from models.lif_neuron import STDPLearner
from utils.reproducibility import set_global_seed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--checkpoint", default="artifacts/checkpoints/nqal_final.pt")
    p.add_argument("--out", default="artifacts/results")
    args = p.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_global_seed(cfg["project"]["seed"])

    nq_cfg = NQALConfig(
        seq_len=cfg["data"]["sequence_length"],
        num_features=cfg["data"]["num_features"],
        num_classes=cfg["data"]["num_classes"],
        hidden_size=cfg["snn"]["num_neurons"],
        use_dsgat=cfg["dsgat"]["num_heads"] > 0,
        dsgat_hidden=cfg["dsgat"]["hidden_dim"],
        dsgat_heads=cfg["dsgat"]["num_heads"],
        seed=cfg["project"]["seed"],
        device="cpu",
    )
    pipe = NQALPipeline(nq_cfg)
    pipe.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])
    pipe.snn.eval()

    # ---- Theorem 1 ----
    X = (torch.rand(8, 50, 4) < 0.3).float()
    thm1 = pipe.qni.verify_theorem1(pipe.snn, X, delta=0.5, num_samples=128)
    print("Theorem 1 (Eq. 10):")
    for k, v in thm1.items():
        print(f"  {k:22s} = {v}")

    # ---- Theorem 2 ----
    stdp = STDPLearner()
    W = torch.rand(nq_cfg.hidden_size, nq_cfg.num_features) * 0.5
    pre = (torch.rand(8, 50, nq_cfg.num_features) < 0.2).float()
    post = (torch.rand(8, 50, nq_cfg.hidden_size) < 0.2).float()
    eff = []
    for step in range(60):
        dw = stdp.trace_update(pre, post, W)
        stdp.apply(W, dw)
        diag = stdp.convergence_diagnostics(W, step)
        eff.append(diag["eff_threshold"])
    eff_np = np.array(eff)
    thm2 = {
        "early_std": float(eff_np[:10].std()),
        "late_std": float(eff_np[-10:].std()),
        "converged": float(eff_np[-10:].std()) <= float(eff_np[:10].std()) + 1e-6,
        "eff_threshold_final": float(eff_np[-1]),
    }
    print("Theorem 2 (Eq. 11):")
    for k, v in thm2.items():
        print(f"  {k:22s} = {v}")

    with open(Path(args.out) / "theorems.json", "w") as f:
        json.dump({"theorem1": thm1, "theorem2": thm2}, f, indent=2)
    print(f"Saved → {args.out}/theorems.json")


if __name__ == "__main__":
    main()
