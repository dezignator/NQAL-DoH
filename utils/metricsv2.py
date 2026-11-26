#!/usr/bin/env python3
"""
utils/metrics.py
================
Metrics suite for NQAL-DoH.

Reproduces Table 2's per-class detection rates and the paper's FPR
target (< 0.5 %).  All functions are NumPy-only so they can be called
from notebooks and the CLI alike.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np


CLASS_NAMES: List[str] = [
    "benign",
    "dns_tunneling",
    "c2_beaconing",
    "dga",
    "adversarial_doh",
]


# -----------------------------------------------------------------------------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


# -----------------------------------------------------------------------------
def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


# -----------------------------------------------------------------------------
def false_positive_rate(
    y_true: np.ndarray, y_pred: np.ndarray, benign_class: int = 0
) -> float:
    """FPR = FP / (FP + TN), where benign is the negative class."""
    benign = y_true == benign_class
    fp = int(((y_pred != benign_class) & benign).sum())
    tn = int(((y_pred == benign_class) & benign).sum())
    return float(fp / max(fp + tn, 1))


# -----------------------------------------------------------------------------
def false_negative_rate(
    y_true: np.ndarray, y_pred: np.ndarray, benign_class: int = 0
) -> float:
    """FNR = FN / (FN + TP) for attack class (any non-benign)."""
    attack = y_true != benign_class
    fn = int(((y_pred == benign_class) & attack).sum())
    tp = int(((y_pred != benign_class) & attack).sum())
    return float(fn / max(fn + tp, 1))


# -----------------------------------------------------------------------------
def per_class_detection(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Per-class recall (== detection rate when the class is the target)."""
    out: Dict[str, float] = {}
    for c, name in enumerate(CLASS_NAMES):
        mask = y_true == c
        if mask.sum() == 0:
            out[name] = 0.0
        else:
            out[name] = float((y_pred[mask] == c).mean())
    return out


# -----------------------------------------------------------------------------
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> float:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        f1s.append(f1)
    return float(np.mean(f1s))


# -----------------------------------------------------------------------------
def adversarial_accuracy(
    model,
    X_adv,
    y_true: np.ndarray,
    batch_size: int = 128,
    device: str = "cpu",
) -> float:
    import torch
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(X_adv), batch_size):
            xb = torch.from_numpy(X_adv[i:i + batch_size]).float().to(device)
            preds.append(model(xb).argmax(-1).cpu().numpy())
    y_pred = np.concatenate(preds)
    return accuracy(y_true, y_pred)


# -----------------------------------------------------------------------------
def composite_score(
    acc: float,
    fpr: float,
    energy_uj: float,
    robustness: float,
) -> float:
    """
    Composite score from Section 3.9.3:

        S = A − F + 1/(1 + log(E_out)) + min_{q∈Q} A(x+q)/A(x)

    Here we substitute E_out = energy_uj (μJ) and robustness = A(x+q)/A(x).
    """
    import math
    e_term = 1.0 / (1.0 + math.log(max(energy_uj, 1e-9)))
    return float(acc - fpr + e_term + robustness)


# -----------------------------------------------------------------------------
def summarize(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy(y_true, y_pred),
        "fpr": false_positive_rate(y_true, y_pred),
        "fnr": false_negative_rate(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        **per_class_detection(y_true, y_pred),
    }
