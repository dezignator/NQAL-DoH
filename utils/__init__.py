"""utils/__init__.py — public utilities."""
from utils.reproducibility import set_global_seed, make_generator, temp_seed
from utils.metrics import (
    accuracy, confusion_matrix, false_positive_rate, false_negative_rate,
    per_class_detection, macro_f1, adversarial_accuracy, composite_score,
    summarize, CLASS_NAMES,
)
from utils.energy import eq9_energy_uj, count_l0, gpu_energy_ratio, energy_report
from utils.logging_utils import setup_logging
from utils.visualization import (
    ensure_dir, plot_robustness_curve, plot_confusion_matrix,
    plot_training_curves, plot_energy_bar,
)

__all__ = [
    "set_global_seed", "make_generator", "temp_seed",
    "accuracy", "confusion_matrix", "false_positive_rate",
    "false_negative_rate", "per_class_detection", "macro_f1",
    "adversarial_accuracy", "composite_score", "summarize", "CLASS_NAMES",
    "eq9_energy_uj", "count_l0", "gpu_energy_ratio", "energy_report",
    "setup_logging",
    "ensure_dir", "plot_robustness_curve", "plot_confusion_matrix",
    "plot_training_curves", "plot_energy_bar",
]
