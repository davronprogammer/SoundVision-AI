"""Comprehensive evaluation and visualization module for SoundVision AI."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from soundvision.data.dataset import SELECTED_CLASSES
from soundvision.evaluation.metrics import compute_classification_metrics


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    class_names: List[str] = SELECTED_CLASSES,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Runs full inference over a dataset, computing comprehensive metrics and per-sample predictions."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_probs: List[np.ndarray] = []
    all_filenames: List[str] = []

    dataset = dataloader.dataset

    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)

            logits = model(data)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs)

    # Collect metadata filenames if available
    if hasattr(dataset, "meta_df"):
        all_filenames = dataset.meta_df["filename"].tolist()
    else:
        all_filenames = [f"sample_{i}.wav" for i in range(len(all_targets))]

    metrics = compute_classification_metrics(all_targets, all_preds, class_names)

    # Build predictions dataframe
    records = []
    for idx in range(len(all_targets)):
        true_idx = all_targets[idx]
        pred_idx = all_preds[idx]
        prob_vec = all_probs[idx]
        confidence = float(prob_vec[pred_idx])

        # Top 3 predictions
        top3_indices = np.argsort(prob_vec)[::-1][:3]
        top3_list = [
            {"class": class_names[j], "probability": float(prob_vec[j])}
            for j in top3_indices
        ]

        records.append({
            "filename": all_filenames[idx],
            "true_idx": true_idx,
            "true_class": class_names[true_idx],
            "pred_idx": pred_idx,
            "predicted_class": class_names[pred_idx],
            "confidence": confidence,
            "is_correct": bool(true_idx == pred_idx),
            "top_3": top3_list,
        })

    preds_df = pd.DataFrame(records)
    return metrics, preds_df


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    output_path: Path,
    title: str = "Confusion Matrix - SoundVision AI",
) -> None:
    """Generates and saves a normalized confusion matrix heatmap."""
    cm_arr = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    norm_cm = np.divide(cm_arr, row_sums, out=np.zeros_like(cm_arr), where=row_sums != 0)

    plt.figure(figsize=(9, 7))
    sns.heatmap(
        norm_cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Recall"},
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11, labelpad=8)
    plt.ylabel("True Class", fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Path,
    title: str = "Training & Validation Metrics",
) -> None:
    """Plots training and validation loss and accuracy curves."""
    epochs = history.get("epoch", list(range(1, len(history["train_loss"]) + 1)))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2, linestyle="--")
    ax1.set_title("Cross-Entropy Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Accuracy plot
    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], label="Train Accuracy", color="#2ca02c", lw=2)
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], label="Val Accuracy", color="#d62728", lw=2, linestyle="--")
    ax2.set_title("Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_per_class_f1(
    per_class_f1: Dict[str, float],
    output_path: Path,
    title: str = "Per-Class F1 Score",
) -> None:
    """Generates a bar chart of F1 score across all classes."""
    classes = list(per_class_f1.keys())
    scores = [per_class_f1[c] * 100 for c in classes]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(classes, scores, color="#3470a3", edgecolor="#1a3d5a")

    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 1.0,
            f"{h:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.ylim(0, 110)
    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Sound Category", fontsize=11, labelpad=8)
    plt.ylabel("F1 Score (%)", fontsize=11, labelpad=8)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_model_comparison(
    models_df: pd.DataFrame,
    output_path: Path,
    title: str = "Experimental Paradigm Comparison",
) -> None:
    """Generates a grouped bar chart comparing models across validation and test metrics."""
    plt.figure(figsize=(10, 5))

    names = models_df["paradigm"].tolist()
    val_accs = [v * 100 if v <= 1.0 else v for v in models_df["best_val_accuracy"].tolist()]

    x = np.arange(len(names))
    width = 0.35

    has_test = "test_accuracy" in models_df.columns and not models_df["test_accuracy"].isna().all()
    if has_test:
        test_accs = [
            (t * 100 if t <= 1.0 else t) if pd.notna(t) else 0.0
            for t in models_df["test_accuracy"].tolist()
        ]
        b1 = plt.bar(x - width / 2, val_accs, width, label="Val Accuracy (Fold 4)", color="#3470a3")
        b2 = plt.bar(x + width / 2, test_accs, width, label="Test Accuracy (Fold 5)", color="#e07a5f")

        for bar in b1:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
        for bar in b2:
            h = bar.get_height()
            if h > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    else:
        b1 = plt.bar(x, val_accs, width, label="Val Accuracy (Fold 4)", color="#3470a3")
        for bar in b1:
            h = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, h + 1, f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Experimental Paradigm", fontsize=11, labelpad=8)
    plt.ylabel("Accuracy (%)", fontsize=11, labelpad=8)
    plt.xticks(x, [n.replace("_", " ").title() for n in names])
    plt.ylim(0, 110)
    plt.legend()
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
