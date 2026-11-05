"""Audio preprocessing pipeline for SoundVision AI.

Handles audio loading, mono conversion, resampling, padding/truncation,
Mel-spectrogram extraction, and z-score normalization.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T


class AudioPreprocessor:
    """Preprocesses raw audio waveforms into normalized multi-channel Mel-spectrograms."""

    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 512,
        n_mels: int = 128,
        f_min: float = 0.0,
        f_max: Optional[float] = 11025.0,
        top_db: float = 80.0,
        target_duration: float = 5.0,
        in_channels: int = 3,
        norm_mean: Optional[float] = None,
        norm_std: Optional[float] = None,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.top_db = top_db
        self.target_duration = target_duration
        self.target_samples = int(sample_rate * target_duration)
        self.in_channels = in_channels
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        # PyTorch transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=self.top_db)

    def set_normalization_stats(self, mean: float, std: float) -> None:
        """Sets the normalization mean and standard deviation computed from the training split."""
        self.norm_mean = float(mean)
        self.norm_std = float(std) if std > 1e-6 else 1.0

    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[torch.Tensor, int]:
        """Loads a WAV audio file, converts to mono, and resamples to target sample rate.

        Raises:
            FileNotFoundError: If audio file does not exist.
            ValueError: If audio file is corrupt or empty.
        """
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            # Using soundfile for robust cross-platform WAV reading
            data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        except Exception as e:
            raise ValueError(f"Corrupt or unreadable audio file '{path}': {e}") from e

        if data.size == 0 or len(data) == 0:
            raise ValueError(f"Empty audio file: {path}")

        # Check for NaN / Inf
        if np.isnan(data).any() or np.isinf(data).any():
            raise ValueError(f"Audio file contains NaN or Inf values: {path}")

        # Convert to mono by averaging channels: shape (samples,)
        mono_data = np.mean(data, axis=1)

        # Convert to torch tensor: shape (1, samples)
        waveform = torch.from_numpy(mono_data).unsqueeze(0)

        # Resample if sample rate differs
        if sr != self.sample_rate:
            resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        return waveform, self.sample_rate

    def adjust_duration(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pads (with reflection or zeros) or center-crops waveform to target duration."""
        num_samples = waveform.shape[-1]
        if num_samples == self.target_samples:
            return waveform

        if num_samples < self.target_samples:
            padding_needed = self.target_samples - num_samples
            # Pad at end with zeros
            return torch.nn.functional.pad(waveform, (0, padding_needed))
        else:
            # Crop to target_samples (center crop)
            start = (num_samples - self.target_samples) // 2
            return waveform[..., start : start + self.target_samples]

    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extracts log-Mel spectrogram in decibels (dB). Shape: (1, n_mels, time_steps)."""
        mel_spec = self.mel_transform(waveform)
        mel_db = self.amplitude_to_db(mel_spec)
        return mel_db

    def normalize(self, mel_db: torch.Tensor) -> torch.Tensor:
        """Applies z-score normalization using training-set statistics."""
        if self.norm_mean is not None and self.norm_std is not None:
            return (mel_db - self.norm_mean) / (self.norm_std + 1e-7)
        return mel_db

    def to_multi_channel(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """Expands 1-channel Mel-spectrogram to multi-channel tensor (e.g. 3 channels for ResNet).

        Input shape: (1, n_mels, time_steps) or (n_mels, time_steps)
        Output shape: (in_channels, n_mels, time_steps)
        """
        if mel_spec.dim() == 2:
            mel_spec = mel_spec.unsqueeze(0)

        if mel_spec.shape[0] == 1 and self.in_channels > 1:
            return mel_spec.repeat(self.in_channels, 1, 1)

        return mel_spec

    def process_waveform(self, waveform: torch.Tensor, apply_norm: bool = True) -> torch.Tensor:
        """Full pipeline from waveform tensor to normalized multi-channel Mel-spectrogram."""
        waveform = self.adjust_duration(waveform)
        mel_db = self.extract_mel_spectrogram(waveform)
        if apply_norm:
            mel_db = self.normalize(mel_db)
        return self.to_multi_channel(mel_db)

    def process_file(self, audio_path: Union[str, Path], apply_norm: bool = True) -> torch.Tensor:
        """Full pipeline from audio file path to normalized multi-channel Mel-spectrogram."""
        waveform, _ = self.load_audio(audio_path)
        return self.process_waveform(waveform, apply_norm=apply_norm)
