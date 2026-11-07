"""Tests for Streamlit application components, audio decoding, and AppTest page workflows."""

import io
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = PROJECT_ROOT / "app"
for p in (str(PROJECT_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pytest
import soundfile as sf
import torch

from components.ui import read_audio
from soundvision.inference.predictor import AudioPredictor
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CKPT_PATH = PROJECT_ROOT / "models" / "final" / "soundvision_resnet18.pth"


def test_read_audio_synthetic_wav():
    """Verifies that synthetic sine wave audio in WAV format decodes correctly."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, sine, sr, format="WAV")
    raw_bytes = buf.getvalue()

    data, decoded_sr = read_audio(raw_bytes, filename="test.wav")
    assert decoded_sr == sr
    assert data.shape[0] == sr
    assert data.ndim == 2  # always_2d returns (samples, channels)


def test_read_audio_synthetic_mp3():
    """Verifies that synthetic sine wave audio in MP3 format decodes correctly."""
    if "MP3" not in sf.available_formats():
        pytest.skip("MP3 format not supported by soundfile environment")

    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, sine, sr, format="MP3")
    raw_bytes = buf.getvalue()

    data, decoded_sr = read_audio(raw_bytes, filename="test.mp3")
    assert decoded_sr == sr
    assert data.shape[0] > 0


def test_read_audio_empty_bytes():
    """Ensures empty or corrupted bytes raise ValueError with clean message."""
    with pytest.raises(ValueError):
        read_audio(b"", filename="empty.wav")

    with pytest.raises(ValueError):
        read_audio(b"RIFF\x00\x00\x00\x00corrupt", filename="corrupt.wav")


def test_predictor_loads_production_checkpoint():
    """Verifies that AudioPredictor loads the 42.7 MB production model without error."""
    assert CKPT_PATH.is_file(), f"Model checkpoint not found at {CKPT_PATH}"
    predictor = AudioPredictor(checkpoint_path=CKPT_PATH, device="cpu")
    assert len(predictor.class_names) == 10
    assert predictor.sample_rate == 22050


def test_edge_cases_short_long_silent():
    """Tests predictor on short (1.5s), long (7.0s), and silent waveforms."""
    predictor = AudioPredictor(checkpoint_path=CKPT_PATH, device="cpu")

    # Short audio
    short_wave = np.random.randn(int(22050 * 1.5)).astype(np.float32)
    res_short = predictor.predict(short_wave, top_k=3)
    assert len(res_short["top_predictions"]) == 3

    # Long audio (triggers sliding window aggregation)
    long_wave = np.random.randn(int(22050 * 7.0)).astype(np.float32)
    res_long = predictor.predict(long_wave, top_k=3)
    assert len(res_long["top_predictions"]) == 3

    # Long audio timeline
    timeline = predictor.predict_timeline(long_wave, window_sec=5.0, hop_sec=2.5)
    assert len(timeline) >= 2

    # Silent audio (all zeros)
    silent_wave = np.zeros(int(22050 * 5.0), dtype=np.float32)
    res_silent = predictor.predict(silent_wave, top_k=1)
    assert res_silent["predicted_class"] in predictor.class_names


def test_apptest_all_four_pages():
    """Runs Streamlit AppTest across Overview, Audio Classifier, Dashboard, and Methodology."""
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception, f"AppTest raised exception on initial load: {at.exception}"

    # Radio widget is at.sidebar.radio[0]
    radio = at.sidebar.radio[0]

    for option in radio.options:
        radio.set_value(option)
        at.run()
        assert not at.exception, f"AppTest raised exception on tab '{option}': {at.exception}"


def test_apptest_bundled_sample_prediction():
    """Simulates picking a bundled demo sample and running prediction in AppTest."""
    at = AppTest.from_file(str(PROJECT_ROOT / "app" / "app.py"), default_timeout=30)
    at.run()

    # Navigate to Audio Classifier
    nav_radio = at.sidebar.radio[0]
    nav_radio.set_value("Audio Classifier")
    at.run()
    assert not at.exception

    # Select "Try Bundled Demo Sample"
    input_radio = at.radio[0]
    input_radio.set_value("Try Bundled Demo Sample")
    at.run()
    assert not at.exception

    # Prediction box should render with predictions
    markdown_texts = [m.value for m in at.markdown]
    has_prediction = any("Predicted Sound Category" in t for t in markdown_texts)
    assert has_prediction, "Prediction result card not found in AppTest output."
