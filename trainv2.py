#!/usr/bin/env python3
"""
train.py
========
End-to-end training for NQAL-DoH.

Stages (matching § 4.2):
    1. Load / generate dataset.
    2. GAN immunisation (Eq. 8).
    3. Centralised adversarial training (Eq. 4).
    4. (Optional) Swarm federated rounds (Eq. 7).
    5. Save checkpoint + metrics.

Usage
-----
    python train.py --config config/default.yaml
    python train.py --config config/default.yaml --smoke-test --max-samples 200 --epochs 1
    python train.py --config config/default.yaml --federated
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import yaml

from models.nqal_pipeline import NQALPipeline, NQALConfig
from utils.reproducibility import set_global_seed
from utils.logging_utils import setup_logging
from utils.metrics import accuracy, summarize
from utils.visualization import plot_training_curves, ensure_dir


# -----------------------------------------------------------------------------
def _load_splits(data_dir: str) -> Dict[str, torch.Tensor]:
    d = Path(data_dir)
    out = {}
    for split in ("train", "val", "test"):
        out[f"X_{split}"] = torch.from_numpy(np.load(d / f"X_{split}.npy")).float()
        out[f"y_{split}"] = torch.from_numpy(np.load(d / f"y_{split}.npy")).long()
    return out


# -----------------------------------------------------------------------------
def _subset(splits, max_samples: int):
    for k in list(splits):
        splits[k] = splits[k][:max_samples]
    return splits


# -----------------------------------------------------------------------------
def make_clients(X, y, num_clients: int, seed: int = 42):
    """Simple balanced split for FL simulation."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    chunks = np.array_split(idx, num_clients)
    return [(X[c], y[c]) for c in chunks]


# -----------------------------------------------------------------------------
@torch.no_grad()
def evaluate_split(pipe: NQALPipeline, X, y, batch_size: int = 128) -> Dict[str, float]:
    pipe.snn.eval()
    preds: List[np.ndarray] = []
    for i in range(0, len(X), batch_size):
        preds.append(pipe.predict(X[i:i + batch_size]).cpu().numpy())
    y_pred = np.concatenate(preds)
    return summarize(y.cpu().numpy(), y_pred)


# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="artifacts/checkpoints")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--federated", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg["project"]["seed"]
    set_global_seed(seed)
    logger = setup_logging()
    ensure_dir(args.output_dir)

    logger.info(f"NQAL-DoH training — seed={seed}")

    # --- Config → NQALConfig ---
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
        qni_lambda_kl=cfg["qni"]["adversarial_loss"]["lambda_kl"],
        qni_epsilon=cfg["qni"]["adversarial_loss"]["epsilon"],
        zkp_tau=cfg["zkp"]["threshold_tau"],
        gan_latent=cfg["gan"]["latent_dim"],
        gan_gamma=cfg["gan"]["adv_weight_gamma"],
        fl_num_clients=cfg["federated"]["num_clients"],
        fl_rounds=cfg["federated"]["communication_rounds"],
        fl_local_epochs=cfg["federated"]["local_epochs"],
        fl_beta=cfg["federated"]["swarm_beta"],
        fl_top_k=cfg["federated"]["compression"]["top_k"],
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        batch_size=cfg["data"]["batch_size"],
        adversarial=cfg["training"]["adversarial_training"]["enabled"],
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    pipe = NQALPipeline(nq_cfg)
    logger.info(f"Pipeline initialised on {nq_cfg.device}")

    # --- Data ---
    splits = _load_splits(args.data_dir)
    if args.max_samples:
        splits = _subset(splits, args.max_samples)
    logger.info(
        f"Data: train={len(splits['X_train'])}, "
        f"val={len(splits['X_val'])}, "
        f"test={len(splits['X_test'])}"
    )

    # --- GAN immunisation (Eq. 8) ---
    if cfg["gan"]["train_steps"] > 0:
        logger.info("Stage 1: GAN immunisation (Eq. 8)")
        pipe.immunize(splits["X_train"], n_steps=cfg["gan"]["train_steps"])

    # --- Training ---
    epochs = args.epochs if args.epochs is not None else cfg["training"]["num_epochs"]
    batch = cfg["data"]["batch_size"]
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
    }

    if args.federated:
        logger.info("Stage 2: Swarm federated training (Eq. 7)")
        clients = make_clients(
            splits["X_train"], splits["y_train"],
            cfg["federated"]["num_clients"], seed=seed,
        )
        for r in range(cfg["federated"]["communication_rounds"]):
            diag = pipe.federated_round(clients)
            val_m = evaluate_split(pipe, splits["X_val"], splits["y_val"])
            history["val_acc"].append(val_m["accuracy"])
            logger.info(
                f"FL round {r+1}/{cfg['federated']['communication_rounds']} "
                f"bytes={diag['total_bytes_sent']} "
                f"val_acc={val_m['accuracy']:.4f}"
            )
    else:
        logger.info("Stage 2: Centralised adversarial training (Eq. 4)")
        n = len(splits["X_train"])
        for epoch in range(epochs):
            pipe.snn.train()
            perm = torch.randperm(n)
            epoch_loss, epoch_correct, epoch_n = 0.0, 0, 0
            t0 = time.perf_counter()
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                diag = pipe.train_step(
                    splits["X_train"][idx], splits["y_train"][idx]
                )
                epoch_loss += diag["loss"] * len(idx)
                with torch.no_grad():
                    preds = pipe.predict(splits["X_train"][idx])
                epoch_correct += int((preds.cpu() == splits["y_train"][idx]).sum())
                epoch_n += len(idx)

            val_m = evaluate_split(pipe, splits["X_val"], splits["y_val"])
            history["train_loss"].append(epoch_loss / epoch_n)
            history["train_acc"].append(epoch_correct / epoch_n)
            history["val_acc"].append(val_m["accuracy"])
            # val_loss computed in eval
            with torch.no_grad():
                logits = pipe.snn(splits["X_val"])
                vl = nn.CrossEntropyLoss()(logits, splits["y_val"]).item()
            history["val_loss"].append(vl)
            dt = time.perf_counter() - t0
            logger.info(
                f"Epoch {epoch+1:02d}/{epochs}  "
                f"train_loss={epoch_loss/epoch_n:.4f}  "
                f"train_acc={epoch_correct/epoch_n:.4f}  "
                f"val_acc={val_m['accuracy']:.4f}  "
                f"val_fpr={val_m['fpr']:.4f}  "
                f"({dt:.1f}s)"
            )
            pipe.epoch = epoch + 1
            # Early stop
            if cfg["training"]["early_stopping"]["enabled"]:
                if len(history["val_acc"]) > cfg["training"]["early_stopping"]["patience"]:
                    best = max(history["val_acc"][:-cfg["training"]["early_stopping"]["patience"]])
                    if history["val_acc"][-1] <= best + 1e-6:
                        logger.info("Early stopping triggered.")
                        break

    # --- Save ---
    ckpt_path = os.path.join(args.output_dir, "nqal_final.pt")
    torch.save(
        {
            "state_dict": pipe.state_dict(),
            "config": cfg,
            "history": history,
            "seed": seed,
        },
        ckpt_path,
    )
    logger.info(f"Checkpoint saved → {ckpt_path}")

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    if history["val_acc"]:
        plot_training_curves(
            history,
            out_path=os.path.join(args.output_dir, "training_curves.png"),
        )
        logger.info("Training curves rendered.")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
