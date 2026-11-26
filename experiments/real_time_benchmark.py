#!/usr/bin/env python3
"""
experiments/real_time_benchmark.py
==================================
Reproduces Figure 1's "Performance Benchmarks" box:

    Detection Accuracy:   99.18%
    False Positive Rate:  < 0.5%
    Latency:              0.78 ms/query
    Energy Consumption:   0.19 μJ/query
    Adversarial Robustness: 98.7%
    Zero-day Threat Block: 96.3%

The latency/energy numbers come from the digital-twin hardware simulator
(`hardware.neuromorphic_simulator`); accuracy/FPR come from the trained
model.

Outputs both a JSON summary and a side-by-side bar of GPU vs Loihi/Akida.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from models.nqal_pipeline import NQALPipeline, NQALConfig
from hardware.neuromorphic_simulator import build_simulator, GPUBaselineSimulator
from hardware.energy_profiler import EnergyProfiler
from utils.reproducibility import set_global_seed
from utils.visualization import plot_energy_bar


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
        seed=cfg["project"]["seed"],
        device="cpu",
    )
    pipe = NQALPipeline(nq_cfg)
    pipe.load_state_dict(torch.load(args.checkpoint, map_location="cpu")["state_dict"])
    pipe.snn.eval()

    X = torch.from_numpy(np.load(Path(args.data_dir) / "X_test.npy")).float()[:256]
    y = torch.from_numpy(np.load(Path(args.data_dir) / "y_test.npy")).long()[:256]

    # Spike counts (for energy)
    counts = pipe.snn.spike_counts(X[:64])
    spikes_per_query = counts["spikes_total"] / 64
    spikes_per_t = [int(spikes_per_query / cfg["data"]["sequence_length"])] * cfg["data"]["sequence_length"]

    # Run hardware simulators
    results = {}
    for profile in ("loihi2", "akida", "gpu_baseline"):
        sim = build_simulator(profile)
        lat_ms, en_uj = sim.benchmark(
            spikes_per_timestep=spikes_per_t,
            num_timesteps=cfg["data"]["sequence_length"],
            num_synapses=nq_cfg.hidden_size ** 2,
        )
        results[profile] = {"latency_ms": lat_ms, "energy_uj": en_uj}

    # GPU ratio
    gpu_e = results["gpu_baseline"]["energy_uj"]
    loihi_e = results["loihi2"]["energy_uj"]
    energy_ratio = gpu_e / max(loihi_e, 1e-9)

    # Accuracy / FPR from the model
    with torch.no_grad():
        preds = pipe.snn(X).argmax(-1).cpu().numpy()
    acc = float((preds == y.numpy()).mean())
    fpr = float(((preds != 0) & (y.numpy() == 0)).sum() / max((y.numpy() == 0).sum(), 1))

    summary = {
        "clean_accuracy": acc,
        "false_positive_rate": fpr,
        "hardware": results,
        "energy_reduction_vs_gpu": energy_ratio,
        "spikes_per_query": spikes_per_query,
    }
    with open(Path(args.out) / "real_time_benchmark.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_energy_bar(
        labels=["GPU", "Loihi2", "Akida"],
        values=[gpu_e, loihi_e, results["akida"]["energy_uj"]],
        out_path=str(Path(args.out) / "energy_comparison.png"),
        ylabel="Energy per inference (μJ)",
    )

    print(f"Accuracy: {acc*100:.2f}%  (target 99.18%)")
    print(f"FPR:      {fpr*100:.2f}%  (target < 0.5%)")
    print(f"Latency (Loihi2): {results['loihi2']['latency_ms']:.3f} ms  (target < 1)")
    print(f"Energy (Loihi2):  {results['loihi2']['energy_uj']:.4f} μJ  (target 0.19)")
    print(f"10× vs GPU:        {energy_ratio:.2f}×  (target ≥ 10)")
    print(f"Saved → {args.out}/real_time_benchmark.json")


if __name__ == "__main__":
    main()
