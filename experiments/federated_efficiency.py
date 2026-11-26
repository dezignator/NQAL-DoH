#!/usr/bin/env python3
"""
experiments/federated_efficiency.py
===================================
Measures federated communication overhead vs. raw-gradient baseline.

Produces:
    * bytes sent per round, for top-k ∈ {1.0, 0.5, 0.1}
    * convergence curve (val accuracy vs. round)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from models.nqal_pipeline import NQALPipeline, NQALConfig
from models.federated import SwarmFederatedLearning, FederatedConfig
from utils.reproducibility import set_global_seed
from utils.metrics import accuracy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--out", default="artifacts/results")
    p.add_argument("--rounds", type=int, default=None)
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

    X_tr = torch.from_numpy(np.load(Path(args.data_dir) / "X_train.npy")).float()[:2000]
    y_tr = torch.from_numpy(np.load(Path(args.data_dir) / "y_train.npy")).long()[:2000]
    X_va = torch.from_numpy(np.load(Path(args.data_dir) / "X_val.npy")).float()[:500]
    y_va = torch.from_numpy(np.load(Path(args.data_dir) / "y_val.npy")).long()[:500]

    # Partitions
    rng = np.random.default_rng(cfg["project"]["seed"])
    idx = rng.permutation(len(X_tr))
    chunks = np.array_split(idx, cfg["federated"]["num_clients"])
    clients = [(X_tr[c], y_tr[c]) for c in chunks]

    criterion = nn.CrossEntropyLoss()
    n_rounds = args.rounds or cfg["federated"]["communication_rounds"]

    rows = []
    for top_k in (1.0, 0.5, 0.1):
        fl = SwarmFederatedLearning(
            global_model=pipe.snn,
            cfg=FederatedConfig(
                num_clients=cfg["federated"]["num_clients"],
                communication_rounds=n_rounds,
                local_epochs=cfg["federated"]["local_epochs"],
                batch_size=cfg["data"]["batch_size"],
                swarm_beta=cfg["federated"]["swarm_beta"],
                top_k=top_k,
                dp_enabled=True,
                seed=cfg["project"]["seed"],
            ),
            qni=pipe.qni,
            device="cpu",
        )
        val_accs = []
        bytes_per_round = []
        for r in range(n_rounds):
            diag = fl.step(clients, criterion, round_idx=r)
            bytes_per_round.append(diag["total_bytes_sent"])
            with torch.no_grad():
                val_pred = pipe.snn(X_va).argmax(-1).cpu().numpy()
            val_accs.append(accuracy(y_va.numpy(), val_pred))
        rows.append({
            "top_k": top_k,
            "mean_bytes_per_round": float(np.mean(bytes_per_round)),
            "final_val_acc": val_accs[-1],
            "convergence": val_accs,
        })
        print(f"top_k={top_k:.1f}: bytes/round={np.mean(bytes_per_round):.0f}, "
              f"val_acc={val_accs[-1]:.4f}")

    # Reduction vs raw (top_k=1.0)
    baseline = rows[0]["mean_bytes_per_round"]
    for r in rows:
        r["reduction_x"] = baseline / r["mean_bytes_per_round"]

    with open(Path(args.out) / "federated_efficiency.json", "w") as f:
        json.dump({
            "num_clients": cfg["federated"]["num_clients"],
            "num_rounds": n_rounds,
            "configs": rows,
        }, f, indent=2)

    with open(Path(args.out) / "federated_efficiency.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["top_k", "mean_bytes_per_round", "final_val_acc", "reduction_x"])
        for r in rows:
            w.writerow([r["top_k"], r["mean_bytes_per_round"], r["final_val_acc"],
                        r["reduction_x"]])

    print(f"Saved → {args.out}/federated_efficiency.{{json,csv}}")


if __name__ == "__main__":
    main()
