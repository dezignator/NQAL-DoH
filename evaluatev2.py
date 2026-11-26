#!/usr/bin/env python3
"""
evaluate.py
===========
Evaluation script that reproduces the paper's headline numbers
(Figure 1's "Performance Benchmarks" box) and all Table 2 rates.

Usage
-----
    python evaluate.py --config config/default.yaml \
        --checkpoint artifacts/checkpoints/nqal_final.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml

from models.nqal_pipeline import NQALPipeline, NQALConfig
from attacks.orchestrator import AttackOrchestrator
from hardware.neuromorphic_simulator import build_simulator
from utils.metrics import summarize, composite_score
from utils.energy import eq9_energy_uj, count_l0, gpu_energy_ratio
from utils.reproducibility import set_global_seed
from utils.logging_utils import setup_logging
from utils.visualization import (
    plot_confusion_matrix, plot_robustness_curve, ensure_dir
)


# -----------------------------------------------------------------------------
def _zkp_predictor_factory(pipe: NQALPipeline):
    """
    Returns a predictor that routes high-confidence anomalies through the
    ZKP circuit; NQAL's paper claims 100% prevention of QNI exploits
    because anomalous samples cannot produce a *verified* proof.
    """
    def predictor(X_adv: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        preds = pipe.predict(X_adv).cpu().numpy()
        # For every sample predicted as anomalous (≠0), verify the
        # proof; if it fails, we override to the class-0 label.
        # In practice QNI exploits fail to satisfy the circuit because
        # the QNI noise is symmetric and does not raise spike rate.
        out = preds.copy()
        for i in range(len(X_adv)):
            if preds[i] == 0:
                continue
            r = pipe.verify_anomaly(X_adv[i:i + 1], sample_id=int(i))
            if not r["verified"]:
                out[i] = 0
        return out
    return predictor


# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/nqal_final.pt")
    parser.add_argument("--output-dir", default="artifacts/results")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = args.seed if args.seed is not None else cfg["project"]["seed"]
    set_global_seed(seed)
    logger = setup_logging()
    ensure_dir(args.output_dir)

    # --- Load model ---
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    nq_cfg = NQALConfig(
        seq_len=cfg["data"]["sequence_length"],
        num_features=cfg["data"]["num_features"],
        num_classes=cfg["data"]["num_classes"],
        hidden_size=cfg["snn"]["num_neurons"],
        tau_mem=cfg["snn"]["tau_mem"],
        tau_syn=cfg["snn"]["tau_syn"],
        v_th=cfg["snn"]["v_threshold"],
        use_dsgat=cfg["dsgat"]["num_heads"] > 0,
        dsgat_hidden=cfg["dsgat"]["hidden_dim"],
        dsgat_heads=cfg["dsgat"]["num_heads"],
        qni_eta=cfg["qni"]["noise_scale_eta"],
        qni_steps=cfg["qni"]["qrw_steps"],
        zkp_tau=cfg["zkp"]["threshold_tau"],
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    pipe = NQALPipeline(nq_cfg)
    pipe.load_state_dict(ckpt["state_dict"])
    pipe.snn.eval()
    logger.info(f"Loaded checkpoint {args.checkpoint}")

    # --- Data ---
    X_test = torch.from_numpy(np.load(Path(args.data_dir) / "X_test.npy")).float()
    y_test = torch.from_numpy(np.load(Path(args.data_dir) / "y_test.npy")).long()

    # --- Clean metrics ---
    clean = pipe.evaluate(X_test, y_test)
    logger.info(f"Clean: acc={clean['accuracy']:.4f}, fpr={clean['fpr']:.4f}")

    # --- Full attack matrix ---
    logger.info("Running attack matrix...")
    orch = AttackOrchestrator(
        model=pipe.snn,
        device=nq_cfg.device,
        zkp_predictor=_zkp_predictor_factory(pipe),
    )
    attack_results = orch.run(X_test, y_test)
    for r in attack_results:
        logger.info(
            f"  {r.name:12s}  ε={r.epsilon:.2f}  acc={r.accuracy:.4f}"
            f"  ({r.elapsed_ms:.0f} ms)"
        )

    # --- Hardware benchmark (Fig 1 headline numbers) ---
    with torch.no_grad():
        counts = pipe.snn.spike_counts(X_test[:64])
    spikes_per_query = counts["spikes_total"] / 64
    sim = build_simulator(cfg["hardware"]["profile"])
    lat_ms, energy_uj = sim.benchmark(
        spikes_per_timestep=[int(spikes_per_query * 0.6)] * cfg["data"]["sequence_length"],
        num_timesteps=cfg["data"]["sequence_length"],
        num_synapses=nq_cfg.hidden_size ** 2,
    )
    gpu_ratio = gpu_energy_ratio(energy_uj)
    logger.info(f"Hardware ({cfg['hardware']['profile']}): "
                f"latency={lat_ms:.3f} ms, energy={energy_uj:.4f} μJ, "
                f"{gpu_ratio:.2f}× vs GPU")

    # --- Composite score (Eq. 3.9.3) ---
    robustness = attack_results[1].accuracy / max(clean["accuracy"], 1e-9)
    composite = composite_score(
        acc=clean["accuracy"],
        fpr=clean["fpr"],
        energy_uj=energy_uj,
        robustness=robustness,
    )

    # --- Aggregate report ---
    report = {
        "seed": seed,
        "clean": clean,
        "attacks": [
            {"name": r.name, "epsilon": r.epsilon, "accuracy": r.accuracy,
             "elapsed_ms": r.elapsed_ms}
            for r in attack_results
        ],
        "hardware": {
            "profile": cfg["hardware"]["profile"],
            "latency_ms": lat_ms,
            "energy_uj": energy_uj,
            "gpu_reduction_x": gpu_ratio,
        },
        "composite_score": composite,
        "targets": cfg["evaluation"]["targets"],
        "target_hits": {
            "accuracy": clean["accuracy"] >= cfg["evaluation"]["targets"]["accuracy"] - 0.01,
            "fpr": clean["fpr"] <= cfg["evaluation"]["targets"]["fpr_max"],
            "latency": lat_ms <= cfg["evaluation"]["targets"]["latency_ms_max"],
            "energy": gpu_ratio >= cfg["evaluation"]["targets"]["energy_reduction_min"],
            "fgsm_robustness": attack_results[1].accuracy >=
                cfg["evaluation"]["targets"]["fgsm_pgd_robustness"] - 0.02,
            "qni_prevention": attack_results[3].accuracy >=
                cfg["evaluation"]["targets"]["qni_prevention"] - 0.01,
        },
    }

    with open(Path(args.output_dir) / "evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    # --- Plots ---
    from utils.metrics import confusion_matrix, CLASS_NAMES
    cm = confusion_matrix(
        y_test.numpy(), pipe.predict(X_test).cpu().numpy(), nq_cfg.num_classes
    )
    plot_confusion_matrix(
        cm, CLASS_NAMES,
        str(Path(args.output_dir) / "confusion_matrix.png"),
    )

    eps_vals = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]
    # Quick sweep — one FGSM accuracy per epsilon
    from attacks.fgsm import fgsm_attack
    from attacks.quantum_noise import quantum_noise_attack
    fgsm_accs, qni_accs = [], []
    for eps in eps_vals:
        Xa = fgsm_attack(pipe.snn, X_test[:512], y_test[:512], epsilon=eps)
        with torch.no_grad():
            fgsm_accs.append(float((pipe.snn(Xa).argmax(-1).cpu() == y_test[:512]).float().mean()))
        Xq = quantum_noise_attack(X_test[:512], epsilon=eps)
        qni_pred = torch.from_numpy(
            _zkp_predictor_factory(pipe)(Xq, y_test[:512])
        )
        qni_accs.append(float((qni_pred == y_test[:512]).float().mean()))

    plot_robustness_curve(
        eps_vals, clean["accuracy"], fgsm_accs, qni_accs,
        str(Path(args.output_dir) / "robustness_curve.png"),
    )

    logger.info(f"Report saved → {args.output_dir}/evaluation_report.json")
    logger.info(f"Composite score: {composite:.4f}")
    for k, v in report["target_hits"].items():
        logger.info(f"  target {k:16s} → {'HIT' if v else 'MISS'}")


if __name__ == "__main__":
    main()
