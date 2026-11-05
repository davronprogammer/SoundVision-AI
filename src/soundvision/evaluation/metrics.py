"""Evaluation metrics calculation for SoundVision AI."""

from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str],
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics given true and predicted class labels."""
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    per_class_f1_vals = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = {
        name: float(per_class_f1_vals[i]) if i < len(per_class_f1_vals) else 0.0
        for i, name in enumerate(class_names)
    }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_str = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm.tolist(),
        "classification_report_dict": report_dict,
        "classification_report_str": report_str,
    }
