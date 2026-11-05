"""Tests for inference API and prediction formatting."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import torch

from soundvision.data.dataset import SELECTED_CLASSES
from soundvision.inference.predictor import AudioPredictor


@pytest.fixture
def mock_checkpoint_path(tmp_path):
    """Creates a mock checkpoint file for testing predictor pipeline."""
    ckpt_path = tmp_path / "mock_resnet18.pth"
    from soundvision.models.resnet import SoundVisionResNet

    model = SoundVisionResNet(num_classes=10, pretrained=False)
    payload = {
        "class_names": SELECTED_CLASSES,
        "sample_rate": 22050,
        "n_fft": 1024,
        "hop_length": 512,
        "n_mels": 128,
        "norm_mean": -30.0,
        "norm_std": 15.0,
        "model_state_dict": model.state_dict(),
    }
    torch.save(payload, ckpt_path)
    return ckpt_path


def test_predictor_wav_file(mock_checkpoint_path, tmp_path):
    predictor = AudioPredictor(checkpoint_path=mock_checkpoint_path, device="cpu")

    # Generate 5-second sine wave audio
    test_wav = tmp_path / "test_tone.wav"
    sr = 22050
    t = np.linspace(0, 5, sr * 5)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(test_wav), audio, sr)

    res = predictor.predict(test_wav, top_k=3)

    assert "predicted_class" in res
    assert res["predicted_class"] in SELECTED_CLASSES
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["top_predictions"]) == 3
    assert res["top_predictions"][0]["class"] == res["predicted_class"]


def test_predictor_variable_duration(mock_checkpoint_path):
    predictor = AudioPredictor(checkpoint_path=mock_checkpoint_path, device="cpu")

    # Short audio (1.5 seconds)
    short_audio = np.random.randn(int(22050 * 1.5)).astype(np.float32)
    res_short = predictor.predict(short_audio, top_k=3)
    assert len(res_short["top_predictions"]) == 3

    # Long audio (9.0 seconds, requires segmentation)
    long_audio = np.random.randn(int(22050 * 9.0)).astype(np.float32)
    res_long = predictor.predict(long_audio, top_k=3)
    assert len(res_long["top_predictions"]) == 3


def test_predictor_missing_file(mock_checkpoint_path):
    predictor = AudioPredictor(checkpoint_path=mock_checkpoint_path, device="cpu")
    with pytest.raises(FileNotFoundError):
        predictor.predict("non_existent_audio_file_12345.wav")
