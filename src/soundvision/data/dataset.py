"""PyTorch Dataset implementation for SoundVision AI (ESC-50 10-class subset).

Strictly isolates folds to avoid data leakage and handles on-the-fly or cached Mel-spectrogram extraction.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from soundvision.data.augmentation import AudioAugmenter
from soundvision.data.preprocessing import AudioPreprocessor

SELECTED_CLASSES: List[str] = [
    "dog",
    "thunderstorm",
    "fireworks",
    "airplane",
    "train",
    "sheep",
    "wind",
    "frog",
    "helicopter",
    "laughing",
]

CLASS_TO_IDX: Dict[str, int] = {cls: idx for idx, cls in enumerate(SELECTED_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {idx: cls for idx, cls in enumerate(SELECTED_CLASSES)}


class ESC50SubsetDataset(Dataset):
    """PyTorch Dataset for the ESC-50 10-category environmental sound benchmark."""

    def __init__(
        self,
        meta_df: pd.DataFrame,
        audio_dir: Union[str, Path],
        preprocessor: AudioPreprocessor,
        augmenter: Optional[AudioAugmenter] = None,
        is_train: bool = False,
        cache_spectrograms: bool = False,
    ):
        """Args:

        meta_df: DataFrame containing metadata filtered for target classes and folds.
        audio_dir: Directory containing raw .wav audio files.
        preprocessor: AudioPreprocessor instance for feature extraction & normalization.
        augmenter: AudioAugmenter instance for training-time augmentations.
        is_train: Flag indicating whether this is a training split (enables augmentation).
        cache_spectrograms: If True, caches raw mel-spectrograms in memory for faster training.
        """
        self.meta_df = meta_df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.preprocessor = preprocessor
        self.augmenter = augmenter if (is_train and augmenter and augmenter.enabled) else None
        self.is_train = is_train
        self.cache_spectrograms = cache_spectrograms
        self._cache: Dict[int, torch.Tensor] = {}
        self._waveform_cache: Dict[int, torch.Tensor] = {}

        # Pre-cache waveforms in RAM to eliminate per-epoch disk I/O overhead
        for i in range(len(self.meta_df)):
            fn = self.meta_df.iloc[i]["filename"]
            p = self.audio_dir / fn
            if p.is_file():
                w, _ = self.preprocessor.load_audio(p)
                self._waveform_cache[i] = self.preprocessor.adjust_duration(w)

    def __len__(self) -> int:
        return len(self.meta_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.meta_df.iloc[idx]
        filename = row["filename"]
        category = row["category"]
        label = CLASS_TO_IDX[category]

        audio_path = self.audio_dir / filename

        # If cache is active and we're not doing waveform augmentation, we can cache mel_db
        if self.cache_spectrograms and idx in self._cache and not self.is_train:
            mel_spec = self._cache[idx].clone()
        else:
            if idx in self._waveform_cache:
                waveform = self._waveform_cache[idx].clone()
            else:
                waveform, _ = self.preprocessor.load_audio(audio_path)
                waveform = self.preprocessor.adjust_duration(waveform)

            # Apply waveform augmentations during training
            if self.is_train and self.augmenter:
                waveform = self.augmenter.augment_waveform(waveform)

            # Extract Mel-spectrogram
            mel_db = self.preprocessor.extract_mel_spectrogram(waveform)

            # Apply spectrogram augmentations (SpecAugment) during training
            if self.is_train and self.augmenter:
                mel_db = self.augmenter.augment_spectrogram(mel_db)

            if self.cache_spectrograms and not self.is_train:
                self._cache[idx] = mel_db.clone()

            mel_spec = mel_db

        # Apply z-score normalization
        mel_spec = self.preprocessor.normalize(mel_spec)

        # Expand to 3 channels for ResNet
        feature = self.preprocessor.to_multi_channel(mel_spec)

        return feature, label

    def get_sample_metadata(self, idx: int) -> Dict[str, Union[str, int]]:
        """Returns metadata for a specific sample."""
        row = self.meta_df.iloc[idx]
        return {
            "filename": row["filename"],
            "category": row["category"],
            "fold": int(row["fold"]),
            "target": CLASS_TO_IDX[row["category"]],
        }


def compute_dataset_norm_stats(
    dataset: ESC50SubsetDataset,
) -> Tuple[float, float]:
    """Computes global mean and standard deviation of Mel-spectrogram dB values

    strictly over the given dataset (must be training split).
    """
    total_sum = 0.0
    total_sq_sum = 0.0
    total_elements = 0

    # Temporarily disable normalization and augmentation for stats calculation
    old_mean = dataset.preprocessor.norm_mean
    old_std = dataset.preprocessor.norm_std
    old_train = dataset.is_train
    dataset.preprocessor.norm_mean = None
    dataset.preprocessor.norm_std = None
    dataset.is_train = False

    try:
        for idx in range(len(dataset)):
            row = dataset.meta_df.iloc[idx]
            audio_path = dataset.audio_dir / row["filename"]
            waveform, _ = dataset.preprocessor.load_audio(audio_path)
            waveform = dataset.preprocessor.adjust_duration(waveform)
            mel_db = dataset.preprocessor.extract_mel_spectrogram(waveform)

            val = mel_db.numpy()
            total_sum += float(np.sum(val))
            total_sq_sum += float(np.sum(val**2))
            total_elements += int(val.size)

        mean = total_sum / total_elements
        variance = (total_sq_sum / total_elements) - (mean**2)
        std = float(np.sqrt(max(variance, 1e-6)))
    finally:
        dataset.preprocessor.norm_mean = old_mean
        dataset.preprocessor.norm_std = old_std
        dataset.is_train = old_train

    return float(mean), float(std)
