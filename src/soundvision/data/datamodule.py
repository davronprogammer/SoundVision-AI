"""DataModule managing dataset loading, fold partitioning, and DataLoaders."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
from torch.utils.data import DataLoader

from soundvision.config import AppConfig
from soundvision.data.augmentation import AudioAugmenter
from soundvision.data.dataset import (
    CLASS_TO_IDX,
    ESC50SubsetDataset,
    SELECTED_CLASSES,
    compute_dataset_norm_stats,
)
from soundvision.data.preprocessing import AudioPreprocessor


class SoundVisionDataModule:
    """Encapsulates dataset preparation, normalization, and DataLoader construction."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.meta_csv_path = Path(config.dataset.meta_csv)
        self.audio_dir = Path(config.dataset.audio_dir)
        self.classes = config.dataset.classes
        self.train_folds = config.dataset.train_folds
        self.val_folds = config.dataset.val_folds
        self.test_folds = config.dataset.test_folds

        self.preprocessor = AudioPreprocessor(
            sample_rate=config.preprocessing.sample_rate,
            n_fft=config.preprocessing.n_fft,
            hop_length=config.preprocessing.hop_length,
            n_mels=config.preprocessing.n_mels,
            f_min=config.preprocessing.f_min,
            f_max=config.preprocessing.f_max,
            top_db=config.preprocessing.top_db,
            target_duration=config.dataset.target_duration_sec,
            in_channels=config.preprocessing.in_channels,
            norm_mean=config.preprocessing.norm_mean,
            norm_std=config.preprocessing.norm_std,
        )

        self.augmenter = AudioAugmenter(
            enabled=config.augmentation.enabled,
            time_mask_max_len=config.augmentation.time_mask_max_len,
            freq_mask_max_len=config.augmentation.freq_mask_max_len,
            num_time_masks=config.augmentation.num_time_masks,
            num_freq_masks=config.augmentation.num_freq_masks,
            gain_range=config.augmentation.gain_range,
            time_shift_max_ratio=config.augmentation.time_shift_max_ratio,
        )

        self.train_dataset: Optional[ESC50SubsetDataset] = None
        self.val_dataset: Optional[ESC50SubsetDataset] = None
        self.test_dataset: Optional[ESC50SubsetDataset] = None
        self.meta_df: Optional[pd.DataFrame] = None
        self.norm_mean: Optional[float] = None
        self.norm_std: Optional[float] = None

    def setup(self, compute_stats: bool = True) -> None:
        """Loads and filters metadata, constructs datasets, and calculates training normalization stats."""
        if not self.meta_csv_path.is_file():
            raise FileNotFoundError(
                f"ESC-50 metadata CSV not found at '{self.meta_csv_path}'. "
                "Run 'python scripts/download_dataset.py' first."
            )

        df = pd.read_csv(self.meta_csv_path)

        # Filter strictly for our 10 classes
        filtered_df = df[df["category"].isin(self.classes)].copy()
        if len(filtered_df) == 0:
            raise ValueError(f"No samples found for classes: {self.classes}")

        self.meta_df = filtered_df

        # Partition into train, validation, and test by fold
        train_df = filtered_df[filtered_df["fold"].isin(self.train_folds)].copy()
        val_df = filtered_df[filtered_df["fold"].isin(self.val_folds)].copy()
        test_df = filtered_df[filtered_df["fold"].isin(self.test_folds)].copy()

        # Build datasets
        self.train_dataset = ESC50SubsetDataset(
            meta_df=train_df,
            audio_dir=self.audio_dir,
            preprocessor=self.preprocessor,
            augmenter=self.augmenter,
            is_train=True,
            cache_spectrograms=False,
        )

        self.val_dataset = ESC50SubsetDataset(
            meta_df=val_df,
            audio_dir=self.audio_dir,
            preprocessor=self.preprocessor,
            augmenter=None,
            is_train=False,
            cache_spectrograms=True,
        )

        self.test_dataset = ESC50SubsetDataset(
            meta_df=test_df,
            audio_dir=self.audio_dir,
            preprocessor=self.preprocessor,
            augmenter=None,
            is_train=False,
            cache_spectrograms=True,
        )

        # Compute or apply normalization stats strictly from train set
        if compute_stats and self.norm_mean is None:
            if self.config.preprocessing.norm_mean is not None and self.config.preprocessing.norm_std is not None:
                self.norm_mean = self.config.preprocessing.norm_mean
                self.norm_std = self.config.preprocessing.norm_std
            else:
                mean, std = compute_dataset_norm_stats(self.train_dataset)
                self.norm_mean = mean
                self.norm_std = std

            self.preprocessor.set_normalization_stats(self.norm_mean, self.norm_std)

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("DataModule not setup. Call `setup()` first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("DataModule not setup. Call `setup()` first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("DataModule not setup. Call `setup()` first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
