"""Tests for audio preprocessing and tensor transformations."""

import numpy as np
import pytest
import soundfile as sf
import torch

from soundvision.data.preprocessing import AudioPreprocessor


def test_audio_duration_adjustment():
    preprocessor = AudioPreprocessor(sample_rate=22050, target_duration=5.0)
    target_samples = 22050 * 5  # 110250

    # Shorter waveform: 2 seconds
    short_wave = torch.randn(1, 22050 * 2)
    adjusted_short = preprocessor.adjust_duration(short_wave)
    assert adjusted_short.shape[-1] == target_samples
    # Verify padding
    assert torch.all(adjusted_short[..., 22050 * 2 :] == 0.0)

    # Longer waveform: 8 seconds
    long_wave = torch.randn(1, 22050 * 8)
    adjusted_long = preprocessor.adjust_duration(long_wave)
    assert adjusted_long.shape[-1] == target_samples


def test_mel_spectrogram_dimensions_and_validity():
    preprocessor = AudioPreprocessor(
        sample_rate=22050,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        target_duration=5.0,
        in_channels=3,
    )
    # Generate 5 seconds of synthetic audio (e.g. sine wave)
    t = torch.linspace(0, 5.0, 22050 * 5)
    waveform = torch.sin(2 * np.pi * 440.0 * t).unsqueeze(0)

    processed = preprocessor.process_waveform(waveform, apply_norm=False)

    # Expected shape: (in_channels=3, n_mels=128, time_steps=216)
    assert processed.dim() == 3
    assert processed.shape[0] == 3
    assert processed.shape[1] == 128
    assert processed.shape[2] == 216

    # Verify absence of NaN or Inf
    assert not torch.isnan(processed).any(), "Mel-spectrogram contains NaNs!"
    assert not torch.isinf(processed).any(), "Mel-spectrogram contains Infs!"


def test_z_score_normalization():
    preprocessor = AudioPreprocessor(
        sample_rate=22050,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        in_channels=1,
    )
    waveform = torch.randn(1, 22050 * 5)
    raw_mel = preprocessor.extract_mel_spectrogram(waveform)

    mean = float(raw_mel.mean().item())
    std = float(raw_mel.std().item())
    preprocessor.set_normalization_stats(mean, std)

    norm_mel = preprocessor.normalize(raw_mel)
    assert pytest.approx(float(norm_mel.mean().item()), abs=1e-2) == 0.0
    assert pytest.approx(float(norm_mel.std().item()), abs=1e-2) == 1.0


def test_empty_audio_detection(tmp_path):
    preprocessor = AudioPreprocessor()
    empty_file = tmp_path / "empty.wav"
    # Create empty WAV file (0 samples)
    with sf.SoundFile(str(empty_file), mode="w", samplerate=22050, channels=1) as f:
        pass

    with pytest.raises(ValueError, match="Empty audio file"):
        preprocessor.load_audio(empty_file)
