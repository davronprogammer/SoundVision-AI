"""Training entry point script for SoundVision AI."""

import argparse
from datetime import datetime
from pathlib import Path
import random
import shutil
from typing import Optional
import numpy as np
import pandas as pd
import torch

from soundvision.config import AppConfig, load_config
from soundvision.data.datamodule import SoundVisionDataModule
from soundvision.evaluation.evaluator import plot_training_curves
from soundvision.models.model_factory import build_model
from soundvision.training.trainer import SoundVisionTrainer


def set_seed(seed: int) -> None:
    """Sets random seeds across standard library, NumPy, and PyTorch for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def log_experiment_to_csv(
    csv_path: Path,
    record: dict,
) -> None:
    """Appends an experiment result entry to the consolidated CSV file."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([record])
    if csv_path.is_file():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, mode="w", header=True, index=False)


def run_training(
    config_path: str = "configs/config.yaml",
    experiment_type: str = "fine_tuning",
    experiment_id: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
) -> dict:
    cfg: AppConfig = load_config(config_path)

    if epochs is not None:
        cfg.training.epochs = epochs
    if batch_size is not None:
        cfg.training.batch_size = batch_size
    if lr is not None:
        cfg.training.lr_head = lr
        cfg.training.lr_backbone = lr * 0.1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not experiment_id:
        experiment_id = f"{experiment_type}_{timestamp}"

    set_seed(cfg.seed)

    print("=" * 60)
    print(f" SOUNDVISION AI — TRAINING PIPELINE [{experiment_id}]")
    print(f" Paradigm: {experiment_type} | Epochs: {cfg.training.epochs} | Batch: {cfg.training.batch_size}")
    print("=" * 60)

    # 1. DataModule setup
    dm = SoundVisionDataModule(cfg)
    dm.setup(compute_stats=True)
    print(f"Computed Train Normalization Stats -> Mean: {dm.norm_mean:.4f}, Std: {dm.norm_std:.4f}")

    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    # 2. Model initialization
    model = build_model(
        experiment_type=experiment_type,
        num_classes=cfg.model.num_classes,
        dropout=cfg.model.dropout,
        in_channels=cfg.preprocessing.in_channels,
    )

    # 3. Trainer
    trainer = SoundVisionTrainer(
        model=model,
        config=cfg,
        experiment_id=experiment_id,
        norm_mean=dm.norm_mean,
        norm_std=dm.norm_std,
    )

    train_res = trainer.fit(train_loader, val_loader)

    # 4. Save training curves
    figures_dir = Path(cfg.paths.figures_dir)
    curve_path = figures_dir / f"training_curves_{experiment_id}.png"
    plot_training_curves(train_res["history"], curve_path, title=f"Training Curves ({experiment_id})")

    # 5. Record experiment metrics in CSV
    best_val_loss = min(train_res["history"]["val_loss"])
    best_val_f1 = max(train_res["history"]["val_f1"])

    experiment_record = {
        "experiment_id": experiment_id,
        "paradigm": experiment_type,
        "architecture": cfg.model.backbone,
        "pretrained": (experiment_type != "scratch"),
        "learning_rate_head": cfg.training.lr_head,
        "learning_rate_backbone": cfg.training.lr_backbone if experiment_type == "fine_tuning" else 0.0,
        "batch_size": cfg.training.batch_size,
        "epochs_trained": len(train_res["history"]["epoch"]),
        "best_val_accuracy": round(train_res["best_val_acc"], 4),
        "best_val_loss": round(best_val_loss, 4),
        "best_val_f1": round(best_val_f1, 4),
        "duration_seconds": round(train_res["training_duration"], 2),
        "checkpoint_path": train_res["best_checkpoint_path"],
    }

    metrics_csv = Path(cfg.paths.metrics_dir) / "experiment_results.csv"
    log_experiment_to_csv(metrics_csv, experiment_record)
    print(f"Logged experiment record to '{metrics_csv}'.")

    # 6. Copy to models/final if fine_tuning or requested
    final_dir = Path(cfg.paths.final_model_dir)
    final_dir.mkdir(parents=True, exist_ok=True)
    if train_res["best_checkpoint_path"] and Path(train_res["best_checkpoint_path"]).is_file():
        final_target = final_dir / "soundvision_resnet18.pth"
        shutil.copy(train_res["best_checkpoint_path"], final_target)
        print(f"Updated final model checkpoint at '{final_target}'.")

    return experiment_record


def main():
    parser = argparse.ArgumentParser(description="Train SoundVision AI model.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--experiment-type",
        type=str,
        default="fine_tuning",
        choices=["scratch", "feature_extraction", "fine_tuning"],
    )
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    run_training(
        config_path=args.config,
        experiment_type=args.experiment_type,
        experiment_id=args.experiment_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
