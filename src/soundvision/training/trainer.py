"""Training engine for SoundVision AI."""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from soundvision.config import AppConfig
from soundvision.data.dataset import IDX_TO_CLASS, SELECTED_CLASSES
from soundvision.evaluation.metrics import compute_classification_metrics
from soundvision.models.resnet import SoundVisionResNet
from soundvision.training.callbacks import EarlyStopping, ModelCheckpoint
from soundvision.training.losses import get_loss_function


class SoundVisionTrainer:
    """Orchestrates model training, validation, early stopping, and checkpointing."""

    def __init__(
        self,
        model: SoundVisionResNet,
        config: AppConfig,
        experiment_id: str,
        norm_mean: Optional[float] = None,
        norm_std: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.config = config
        self.experiment_id = experiment_id
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model.to(self.device)

        # Loss function
        self.criterion = get_loss_function(label_smoothing=config.training.label_smoothing)

        # Optimizer with discriminative learning rates
        param_groups = self.model.get_parameter_groups(
            lr_backbone=config.training.lr_backbone,
            lr_head=config.training.lr_head,
            weight_decay=config.training.weight_decay,
        )
        self.optimizer = torch.optim.AdamW(param_groups)

        # LR Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs,
            eta_min=1e-6,
        )

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            mode="max",
        )

        checkpoints_dir = Path(config.paths.checkpoints_dir) / experiment_id
        self.checkpoint_cb = ModelCheckpoint(
            checkpoint_dir=checkpoints_dir,
            filename_prefix=experiment_id,
            monitor="val_acc",
            mode="max",
        )

        # Automatic mixed precision
        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "train_acc": [],
            "train_f1": [],
            "val_loss": [],
            "val_acc": [],
            "val_f1": [],
            "lr": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []

        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping
            if self.config.training.gradient_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.gradient_clip_norm
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * data.size(0)
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.detach().cpu().numpy())

        epoch_loss = total_loss / len(train_loader.dataset)
        metrics = compute_classification_metrics(all_targets, all_preds, SELECTED_CLASSES)

        return {
            "loss": epoch_loss,
            "acc": metrics["accuracy"],
            "f1": metrics["macro_f1"],
        }

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Tuple[Dict[str, float], Dict[str, Any]]:
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_targets: List[int] = []

        for data, targets in val_loader:
            data = data.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

            total_loss += loss.item() * data.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

        epoch_loss = total_loss / len(val_loader.dataset)
        full_metrics = compute_classification_metrics(all_targets, all_preds, SELECTED_CLASSES)

        summary_metrics = {
            "loss": epoch_loss,
            "acc": full_metrics["accuracy"],
            "f1": full_metrics["macro_f1"],
        }
        return summary_metrics, full_metrics

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Runs complete training loop over configured epochs."""
        start_time = time.time()
        print(f"--- Starting Training [{self.experiment_id}] on {self.device} ---")

        for epoch in range(1, self.config.training.epochs + 1):
            ep_start = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics, val_full = self.evaluate(val_loader)

            current_lr = self.optimizer.param_groups[-1]["lr"]
            self.scheduler.step()

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["acc"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["acc"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["lr"].append(current_lr)

            # Checkpoint payload
            checkpoint_payload = {
                "experiment_id": self.experiment_id,
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_acc": val_metrics["acc"],
                "val_f1": val_metrics["f1"],
                "val_loss": val_metrics["loss"],
                "class_names": SELECTED_CLASSES,
                "sample_rate": self.config.preprocessing.sample_rate,
                "n_fft": self.config.preprocessing.n_fft,
                "hop_length": self.config.preprocessing.hop_length,
                "n_mels": self.config.preprocessing.n_mels,
                "norm_mean": self.norm_mean,
                "norm_std": self.norm_std,
                "preprocessing_config": vars(self.config.preprocessing),
                "model_config": vars(self.config.model),
            }

            saved = self.checkpoint_cb.step(val_metrics["acc"], checkpoint_payload, epoch)
            mark = " *BEST*" if saved else ""

            duration = time.time() - ep_start
            print(
                f"Epoch {epoch:02d}/{self.config.training.epochs:02d} [{duration:.1f}s] - "
                f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}, F1: {train_metrics['f1']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}{mark}",
                flush=True,
            )

            if self.early_stopping(val_metrics["acc"]):
                print(f"Early stopping triggered at epoch {epoch}.", flush=True)
                break

        total_duration = time.time() - start_time
        print(f"--- Training Completed in {total_duration:.1f}s ---", flush=True)

        return {
            "best_checkpoint_path": str(self.checkpoint_cb.best_checkpoint_path),
            "best_val_acc": self.checkpoint_cb.best_metric,
            "history": self.history,
            "training_duration": total_duration,
        }
