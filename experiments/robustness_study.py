#!/usr/bin/env python3
"""
experiments/robustness_study.py
===============================
Reproduces the paper's headline robustness numbers:
    * 94% robustness against FGSM/PGD
    * 100% prevention of QNI exploits

Sweeps ε and writes a JSON + CSV + robustness curve PNG.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from models.nqal_pipeline import NQALPipeline, NQALConfig
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.quantum_noise import quantum_noise_attack
from utils.reproducibility import set_global_seed
from utils.visualization import plot_robustness_curve


def _zkp_predictor(pipe):
    def predictor(X_adv, y):
        preds = pipe.predict(X_adv).cpu().numpy()
        out = preds.copy()
        for i in range(len(X_adv)):
            if preds[i] == 0:
                continue
            r = pipe.verify_anomaly(X_adv[i:i + 1], sample_id=int(i))
            if not r["verified"]:
                out[i] = 0
        return out
    return predictor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/default.yaml")
    p.add_argument("--data-dir", default="data/processed")
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
        zkp_tau=cfg["zkp"]["threshold_tau"],
        seed=cfg["project"]["seed"],
        device="cpu",
    )
    pipe = NQALPipeline(nq_cfg)
    pipe.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])
    pipe.snn.eval()

    X = torch.from_numpy(np.load(Path(args.data_dir) / "X_test.npy")).float()[:1024]
    y = torch.from_numpy(np.load(Path(args.data_dir) / "y_test.npy")).long()[:1024]

    with torch.no_grad():
        clean_acc = float((pipe.snn(X).argmax(-1) == y).float().mean())
    print(f"Clean accuracy: {clean_acc:.4f}")

    eps_grid = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    rows = []
    fgsm_accs, pgd_accs, qni_accs = [], [], []
    zkp = _zkp_predictor(pipe)

    for eps in eps_grid:
        Xf = fgsm_attack(pipe.snn, X, y, epsilon=eps)
        with torch.no_grad():
            a_f = float((pipe.snn(Xf).argmax(-1) == y).float().mean())
        Xp = pgd_attack(pipe.snn, X, y, epsilon=eps, steps=10)
        with torch.no_grad():
            a_p = float((pipe.snn(Xp).argmax(-1) == y).float().mean())
        Xq = quantum_noise_attack(X, epsilon=eps)
        a_q = float((torch.from_numpy(zkp(Xq, y)) == y).float().mean())
        fgsm_accs.append(a_f); pgd_accs.append(a_p); qni_accs.append(a_q)
        rows.append({"epsilon": eps, "fgsm": a_f, "pgd": a_p, "qni_zkp": a_q})
        print(f"  ε={eps:.2f}: FGSM={a_f:.4f}, PGD={a_p:.4f}, QNI(ZKP)={a_q:.4f}")

    # Averages for the paper's headline numbers
    avg_fgsm_pgd = float(np.mean([r["fgsm"] for r in rows] + [r["pgd"] for r in rows]))
    avg_qni = float(np.mean([r["qni_zkp"] for r in rows]))
    print(f"Mean FGSM+PGD robustness: {avg_fgsm_pgd:.4f}  (paper target 0.94)")
    print(f"Mean QNI prevention:      {avg_qni:.4f}  (paper target 1.00)")

    with open(Path(args.out) / "robustness_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    with open(Path(args.out) / "robustness_summary.json", "w") as f:
        json.dump({
            "clean_accuracy": clean_acc,
            "eps_grid": eps_grid,
            "mean_fgsm_pgd_accuracy": avg_fgsm_pgd,
            "mean_qni_zkp_accuracy": avg_qni,
            "paper_targets": {"fgsm_pgd": 0.94, "qni": 1.00},
        }, f, indent=2)

    # Plot FGSM vs QNI
    plot_robustness_curve(
        eps_grid, clean_acc, fgsm_accs, qni_accs,
        str(Path(args.out) / "robustness_curve.png"),
    )
    print(f"Saved → {args.out}/robustness_*.{{csv,json,png}}")


if __name__ == "__main__":
    main()
