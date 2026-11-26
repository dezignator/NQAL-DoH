"""
Evaluation metrics: accuracy, FPR, detection rate.
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

def compute_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def compute_fpr(y_true, y_pred, positive_class=0):
    """False Positive Rate for benign class."""
    cm = confusion_matrix(y_true, y_pred)
    # Assuming class 0 is benign
    fp = cm[0, 1:].sum() if cm.shape[0] > 1 else 0
    tn = cm[0, 0]
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return fpr

def compute_detection_rate(y_true, y_pred, attack_class):
    """Detection rate for a specific attack class."""
    cm = confusion_matrix(y_true, y_pred)
    if attack_class >= cm.shape[0]:
        return 0.0
    tp = cm[attack_class, attack_class]
    fn = cm[attack_class, :].sum() - tp
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0
