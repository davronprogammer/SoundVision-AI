"""Generates comprehensive comparative visualization for SoundVision AI experiments."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from soundvision.config import load_config


def generate_comparison_plot(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    metrics_dir = Path(cfg.paths.metrics_dir)
    figures_dir = Path(cfg.paths.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    exp_csv = metrics_dir / "experiment_results.csv"
    if not exp_csv.is_file():
        print(f"No experiment_results.csv found at {exp_csv}")
        return

    df = pd.read_csv(exp_csv)
    print("Experiments loaded:")
    print(df[["experiment_id", "paradigm", "best_val_accuracy", "best_val_f1"]])

    # Historical vs reproduced mapping
    paradigms = ["scratch", "feature_extraction", "fine_tuning"]
    labels = ["ResNet-18\n(From Scratch)", "ResNet-18\n(Feature Extraction)", "ResNet-18\n(Fine-Tuning)"]

    val_accs = []
    val_f1s = []

    for p in paradigms:
        rows = df[df["paradigm"] == p]
        if len(rows) > 0:
            latest = rows.iloc[-1]
            val_accs.append(latest["best_val_accuracy"] * 100)
            val_f1s.append(latest["best_val_f1"] * 100)
        else:
            val_accs.append(0.0)
            val_f1s.append(0.0)

    # Historical colab metrics
    historical_val_acc = [61.25, 12.50, 86.25]
    historical_test_acc = [46.25, 10.00, 70.00]

    # Create figure
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    x = np.arange(len(paradigms))
    width = 0.35

    # Plot 1: Historical Colab Validation vs Reproduced Validation
    rects1 = ax1.bar(x - width/2, historical_val_acc, width, label="Historical Colab Val Acc (%)", color="#90cdf4", edgecolor="#2b6cb0")
    rects2 = ax1.bar(x + width/2, val_accs, width, label="Reproduced Local Val Acc (%)", color="#2b6cb0", edgecolor="#1a365d")

    ax1.set_ylabel("Accuracy (%)", fontsize=11, fontweight="bold")
    ax1.set_title("Validation Accuracy: Historical vs. Reproduced", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Bar value labels
    for rect in rects1:
        h = rect.get_height()
        if h > 0:
            ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects2:
        h = rect.get_height()
        if h > 0:
            ax1.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Plot 2: Reproduced Best Validation Acc vs Macro F1
    rects3 = ax2.bar(x - width/2, val_accs, width, label="Best Val Accuracy (%)", color="#319795", edgecolor="#234e52")
    rects4 = ax2.bar(x + width/2, val_f1s, width, label="Best Val Macro F1 (%)", color="#805ad5", edgecolor="#44337a")

    ax2.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
    ax2.set_title("Reproduced Performance: Validation Accuracy vs. Macro F1", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylim(0, 100)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.6)

    for rect in rects3:
        h = rect.get_height()
        if h > 0:
            ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in rects4:
        h = rect.get_height()
        if h > 0:
            ax2.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_file = figures_dir / "model_comparison.png"
    fig.savefig(out_file, dpi=300)
    plt.close(fig)
    print(f"Generated comparison plot saved to: '{out_file}'")


if __name__ == "__main__":
    generate_comparison_plot()
