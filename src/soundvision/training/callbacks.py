"""Callbacks for training: EarlyStopping and ModelCheckpoint."""

from pathlib import Path
from typing import Any, Dict, Optional
import torch


class EarlyStopping:
    """Early stops the training if validation metric does not improve after a given patience."""

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "max":
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return self.early_stop


class ModelCheckpoint:
    """Saves model weights when the monitored validation metric improves."""

    def __init__(
        self,
        checkpoint_dir: Path,
        filename_prefix: str = "best_model",
        monitor: str = "val_acc",
        mode: str = "max",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.filename_prefix = filename_prefix
        self.monitor = monitor
        self.mode = mode
        self.best_metric: Optional[float] = None
        self.best_checkpoint_path: Optional[Path] = None

    def step(self, current_metric: float, checkpoint_payload: Dict[str, Any], epoch: int) -> bool:
        """Saves checkpoint if metric improved. Returns True if a new best was saved."""
        if self.best_metric is None:
            improved = True
        elif self.mode == "max":
            improved = current_metric > self.best_metric
        else:
            improved = current_metric < self.best_metric

        if improved:
            self.best_metric = current_metric
            save_path = self.checkpoint_dir / f"{self.filename_prefix}_epoch_{epoch:02d}.pth"
            best_symlink_path = self.checkpoint_dir / f"{self.filename_prefix}.pth"

            torch.save(checkpoint_payload, save_path)
            torch.save(checkpoint_payload, best_symlink_path)
            self.best_checkpoint_path = best_symlink_path
            return True
        return False
