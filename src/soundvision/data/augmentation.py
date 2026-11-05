"""Data augmentation module for SoundVision AI.

Includes waveform-level augmentations (random gain, time shift, Gaussian noise)
and spectrogram-level augmentations (SpecAugment frequency and time masking).
Augmentations are designed to be applied strictly during training.
"""

from typing import List, Optional
import torch
import torchaudio.transforms as T


class AudioAugmenter:
    """Configurable audio and spectrogram augmentation pipeline for training."""

    def __init__(
        self,
        enabled: bool = True,
        time_mask_max_len: int = 24,
        freq_mask_max_len: int = 16,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        gain_range: Optional[List[float]] = None,
        time_shift_max_ratio: float = 0.1,
    ):
        self.enabled = enabled
        self.time_mask_max_len = time_mask_max_len
        self.freq_mask_max_len = freq_mask_max_len
        self.num_time_masks = num_time_masks
        self.num_freq_masks = num_freq_masks
        self.gain_range = gain_range or [-3.0, 3.0]
        self.time_shift_max_ratio = time_shift_max_ratio

        # PyTorch Torchaudio SpecAugment transforms
        self.time_maskers = [
            T.TimeMasking(time_mask_param=self.time_mask_max_len)
            for _ in range(self.num_time_masks)
        ]
        self.freq_maskers = [
            T.FrequencyMasking(freq_mask_param=self.freq_mask_max_len)
            for _ in range(self.num_freq_masks)
        ]

    def augment_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """Applies waveform-level random gain and time shifting.

        Input shape: (..., samples)
        """
        if not self.enabled:
            return waveform

        # 1. Random Gain (dB)
        min_gain_db, max_gain_db = self.gain_range
        gain_db = torch.empty(1).uniform_(min_gain_db, max_gain_db).item()
        gain_factor = 10.0 ** (gain_db / 20.0)
        waveform = waveform * gain_factor

        # 2. Random Time Shifting (circular shift)
        if self.time_shift_max_ratio > 0.0:
            num_samples = waveform.shape[-1]
            max_shift = int(num_samples * self.time_shift_max_ratio)
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            waveform = torch.roll(waveform, shifts=shift, dims=-1)

        # 3. Very subtle additive Gaussian noise (SNR > 30 dB)
        noise = torch.randn_like(waveform) * 0.001
        waveform = waveform + noise

        return waveform

    def augment_spectrogram(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Applies SpecAugment: random frequency masking and time masking.

        Input shape: (channels, n_mels, time_steps) or (1, n_mels, time_steps)
        """
        if not self.enabled:
            return mel_spec

        # SpecAugment operates on (..., freq, time)
        augmented = mel_spec.clone()

        # Apply frequency masking
        for f_mask in self.freq_maskers:
            augmented = f_mask(augmented)

        # Apply time masking
        for t_mask in self.time_maskers:
            augmented = t_mask(augmented)

        return augmented
