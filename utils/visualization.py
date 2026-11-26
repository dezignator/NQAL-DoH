#!/usr/bin/env python3
"""utils/visualization.py — Figure rendering helpers."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def plot_robustness_curve(
    eps_values: List[float],
    accuracy_clean: float,
    accuracy_adv: List[float],
    accuracy_qni: List[float],
    out_path: str,
    title: str = "Adversarial robustness (FGSM / PGD / QNI)",
) -> str:
    plt.figure(figsize=(7, 4.5))
    plt.plot(eps_values, [accuracy_clean] * len(eps_values),
             "--", label="Clean", color="black")
    plt.plot(eps_values, accuracy_adv, "-o",
             label="FGSM/PGD", color=PALETTE[0])
    plt.plot(eps_values, accuracy_qni, "-s",
             label="QNI exploit", color=PALETTE[2])
    plt.axhline(0.94, color="gray", ls=":", lw=1,
                label="Paper target (94%)")
    plt.xlabel("Perturbation ε")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    out_path: str,
) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_training_curves(
    history: Dict[str, List[float]],
    out_path: str,
) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, color in [("train_loss", PALETTE[0]), ("val_loss", PALETTE[3])]:
        if key in history:
            axes[0].plot(history[key], label=key, color=color)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    for key, color in [("train_acc", PALETTE[2]), ("val_acc", PALETTE[4])]:
        if key in history:
            axes[1].plot(history[key], label=key, color=color)
    axes[1].axhline(0.9918, color="gray", ls=":", label="Paper 99.18%")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_energy_bar(
    labels: List[str], values: List[float], out_path: str,
    ylabel: str = "Energy (μJ)",
) -> str:
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=PALETTE[:len(labels)])
    plt.ylabel(ylabel)
    for i, v in enumerate(values):
        plt.text(i, v, f"{v:.3g}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
