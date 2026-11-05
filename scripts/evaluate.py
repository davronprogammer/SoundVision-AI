"""Evaluation entry point script for SoundVision AI."""

import argparse
import json
from pathlib import Path
from typing import Optional
import pandas as pd
import torch

from soundvision.config import AppConfig, load_config
from soundvision.data.datamodule import SoundVisionDataModule
from soundvision.data.dataset import SELECTED_CLASSES
from soundvision.evaluation.error_analysis import generate_error_markdown, perform_error_analysis
from soundvision.evaluation.evaluator import (
    evaluate_model,
    plot_confusion_matrix,
    plot_per_class_f1,
)
from soundvision.models.resnet import SoundVisionResNet


def run_evaluation(
    checkpoint_path: str = "models/final/soundvision_resnet18.pth",
    config_path: str = "configs/config.yaml",
    split: str = "test",
) -> dict:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at '{ckpt_path}'")

    cfg: AppConfig = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f" SOUNDVISION AI — EVALUATION PIPELINE")
    print(f" Checkpoint: {ckpt_path} | Split: {split} | Device: {device}")
    print("=" * 60)

    # 1. Load Checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_names = checkpoint.get("class_names", SELECTED_CLASSES)
    norm_mean = checkpoint.get("norm_mean")
    norm_std = checkpoint.get("norm_std")

    # Set norm stats in config if available
    if norm_mean is not None and norm_std is not None:
        cfg.preprocessing.norm_mean = norm_mean
        cfg.preprocessing.norm_std = norm_std

    # 2. Reconstruct Model
    model = SoundVisionResNet(
        num_classes=len(class_names),
        pretrained=False,
        freeze_backbone=False,
        dropout=0.0,
        in_channels=cfg.preprocessing.in_channels,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 3. Setup DataModule
    dm = SoundVisionDataModule(cfg)
    dm.setup(compute_stats=False)

    if split == "test":
        loader = dm.test_dataloader()
    elif split == "val":
        loader = dm.val_dataloader()
    else:
        loader = dm.train_dataloader()

    # 4. Run Evaluation
    metrics, preds_df = evaluate_model(model, loader, class_names=class_names, device=device)

    print("\n--- TEST CLASSIFICATION REPORT ---")
    print(metrics["classification_report_str"])
    print(f"Overall Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1 Score:   {metrics['macro_f1'] * 100:.2f}%")

    # 5. Save Artifacts & Figures
    figures_dir = Path(cfg.paths.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(cfg.paths.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    preds_dir = Path(cfg.paths.predictions_dir)
    preds_dir.mkdir(parents=True, exist_ok=True)

    # Plots
    cm_path = figures_dir / f"confusion_matrix_{split}.png"
    plot_confusion_matrix(metrics["confusion_matrix"], class_names, cm_path)

    f1_path = figures_dir / f"per_class_f1_{split}.png"
    plot_per_class_f1(metrics["per_class_f1"], f1_path)

    # Predictions CSV
    preds_csv = preds_dir / f"predictions_{split}.csv"
    preds_df.to_csv(preds_csv, index=False)

    # Metrics JSON
    json_metrics = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "per_class_f1": metrics["per_class_f1"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
    with open(metrics_dir / f"{split}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(json_metrics, f, indent=2)

    # 6. Error Analysis
    error_csv = preds_dir / "misclassified_samples.csv"
    analysis_res = perform_error_analysis(preds_df, output_csv_path=error_csv)
    error_md = generate_error_markdown(analysis_res, model_name=f"SoundVision ResNet18 ({split})")

    with open("docs/ERROR_ANALYSIS.md", "w", encoding="utf-8") as f:
        f.write(error_md)

    print(f"\nSaved evaluation outputs:")
    print(f"  - Confusion Matrix:  {cm_path}")
    print(f"  - Per-Class F1 Plot: {f1_path}")
    print(f"  - Predictions CSV:   {preds_csv}")
    print(f"  - Error Report:      docs/ERROR_ANALYSIS.md")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate SoundVision AI model.")
    parser.add_argument("--checkpoint", type=str, default="models/final/soundvision_resnet18.pth")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--split", type=str, default="test", choices=["test", "val", "train"])
    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        split=args.split,
    )


if __name__ == "__main__":
    main()
