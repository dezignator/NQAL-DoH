#!/usr/bin/env python3
"""
experiments/zkp_privacy_audit.py
================================
Audits the ZKP-Learn module (§ 3.4):
    * proof size vs raw packet bytes
    * proof / verify latency
    * completeness (honest proofs verify)
    * soundness (tampered proofs rejected)
    * Fiat-Shamir binding (wrong sample_id rejected)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from models.nqal_pipeline import NQALPipeline, NQALConfig
from models.zkp import ZKPLearn, ZKProof
from py_ecc.bn128 import multiply
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

    zkp = ZKPLearn(threshold_tau=cfg["zkp"]["threshold_tau"], seed=cfg["project"]["seed"])

    # Synthetic anomalous spike trains (high rate)
    anomalous = (torch.rand(50, 4) < 0.7).float()
    proof = zkp.prove(anomalous, sample_id=0)

    audit = zkp.privacy_audit(anomalous, sample_id=0)

    # Soundness: tamper
    tampered = ZKProof(
        pi_a=proof.pi_a, pi_b=proof.pi_b,
        pi_c=multiply(proof.pi_c, 2),
        public_input_hash=proof.public_input_hash,
        sample_id=proof.sample_id,
    )
    tamper_ok = not zkp.verify(tampered, sample_id=0)

    # Fiat-Shamir binding: wrong sample_id
    binding_ok = not zkp.verify(proof, sample_id=1)

    bench = zkp.benchmark(n_iter=20)

    summary = {
        "proof_size_bytes": audit["proof_size_bytes"],
        "raw_input_bytes": audit["raw_input_size_bytes"],
        "compression_ratio": audit["compression_ratio"],
        "raw_data_exposed": audit["raw_data_exposed"],
        "spike_train_exposed": audit["spike_train_exposed"],
        "completeness": audit["verified"],
        "soundness_tamper_rejected": tamper_ok,
        "fiat_shamir_binding": binding_ok,
        "prove_ms": bench["prove_ms"],
        "verify_ms": bench["verify_ms"],
    }
    with open(Path(args.out) / "zkp_privacy_audit.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Saved → {args.out}/zkp_privacy_audit.json")


if __name__ == "__main__":
    main()
