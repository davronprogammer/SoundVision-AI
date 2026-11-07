"""UI styling, audio decoding, and visualization helpers for SoundVision AI Streamlit app."""

import io
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import torch

from soundvision.data.preprocessing import AudioPreprocessor

CLASS_ICONS: Dict[str, str] = {
    "dog": "🐕",
    "thunderstorm": "⛈️",
    "fireworks": "🎆",
    "airplane": "✈️",
    "train": "🚆",
    "sheep": "🐑",
    "wind": "💨",
    "frog": "🐸",
    "helicopter": "🚁",
    "laughing": "😂",
}

CUSTOM_CSS = """
<style>
    /* Clean, professional styling with adaptive contrast */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a365d;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    .metric-card {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2b6cb0;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        margin-top: 0.2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #ebf8ff 0%, #edf2f7 100%);
        border: 1px solid #bee3f8;
        border-radius: 10px;
        padding: 1.25rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .class-badge {
        display: inline-block;
        background-color: #edf2f7;
        color: #2d3748;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 2px;
    }
    .status-badge {
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .status-badge-ok {
        background-color: #c6f6d5;
        color: #22543d;
    }
    .status-badge-warn {
        background-color: #feebc8;
        color: #744210;
    }
</style>
"""


def apply_custom_css():
    """Injects custom CSS stylesheets into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def read_audio(audio_bytes: bytes, filename: str = "") -> Tuple[np.ndarray, int]:
    """Decodes audio bytes supporting WAV and MP3 formats via soundfile and librosa fallback.

    Args:
        audio_bytes: Raw bytes from uploaded file or local audio recording.
        filename: Optional filename hint to assist codec selection.

    Returns:
        Tuple of (audio_array, sample_rate) where audio_array is float32 of shape (n_samples, n_channels).

    Raises:
        ValueError: If audio buffer is empty, corrupt, or in an unreadable format.
    """
    if not audio_bytes or len(audio_bytes) < 32:
        raise ValueError("Audio buffer is empty or contains insufficient data.")

    # Primary strategy: libsndfile C-based decoder (soundfile)
    try:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
        if data.size > 0:
            return data, sr
    except Exception:
        pass

    # Secondary strategy: librosa decoder with temp file
    suffix = Path(filename).suffix if filename else ".wav"
    if not suffix or suffix.lower() not in [".wav", ".mp3", ".ogg", ".flac"]:
        suffix = ".wav"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        y, sr = librosa.load(tmp_path, sr=None, mono=False)
        if y.ndim == 1:
            data = y[:, np.newaxis].astype(np.float32)
        else:
            data = y.T.astype(np.float32)
        return data, sr
    except Exception as e:
        raise ValueError(
            f"Unable to decode audio data. Please ensure your file is a valid WAV or MP3 recording. (Error: {e})"
        ) from e
    finally:
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except OSError:
                pass


def plot_waveform_and_spectrogram(
    waveform: np.ndarray,
    sr: int,
    mel_db: np.ndarray,
) -> plt.Figure:
    """Creates a stacked waveform and Mel-spectrogram visualization."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5), sharex=False)

    # Waveform
    time_axis = np.linspace(0, len(waveform) / sr, num=len(waveform))
    ax1.plot(time_axis, waveform, color="#2b6cb0", lw=0.8)
    ax1.set_title("Audio Waveform", fontsize=10, fontweight="bold", color="#1a202c")
    ax1.set_ylabel("Amplitude", fontsize=9, color="#2d3748")
    ax1.set_xlim(0, max(time_axis[-1], 0.1))
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Mel-spectrogram
    ax2.imshow(
        mel_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, max(time_axis[-1], 0.1), 0, sr // 2],
    )
    ax2.set_title("Log-Mel Spectrogram (dB)", fontsize=10, fontweight="bold", color="#1a202c")
    ax2.set_xlabel("Time (s)", fontsize=9, color="#2d3748")
    ax2.set_ylabel("Frequency (Hz)", fontsize=9, color="#2d3748")

    plt.tight_layout()
    return fig


def plot_top_predictions_bar(top_predictions: List[Dict[str, Any]]) -> plt.Figure:
    """Plots horizontal bar chart of top candidate probabilities."""
    classes = [item["class"].capitalize() for item in reversed(top_predictions)]
    probs = [item["probability"] * 100 for item in reversed(top_predictions)]

    fig, ax = plt.subplots(figsize=(7, max(1.8, len(classes) * 0.6)))
    bars = ax.barh(classes, probs, color="#3182ce", height=0.55, edgecolor="#2b6cb0")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Confidence (%)", fontsize=9, fontweight="bold", color="#2d3748")

    # Add numeric labels at end of each bar
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold",
            color="#2d3748",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e0")
    ax.spines["bottom"].set_color("#cbd5e0")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    plt.tight_layout()
    return fig


def plot_timeline(timeline_data: List[Dict[str, Any]]) -> plt.Figure:
    """Plots temporal progression of class predictions over sliding analysis windows."""
    if not timeline_data:
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(0.5, 0.5, "No window data", ha="center", va="center")
        return fig

    times = [f"{w['start_sec']:.1f}-{w['end_sec']:.1f}s" for w in timeline_data]
    confs = [w["confidence"] * 100 for w in timeline_data]
    labels = [f"{CLASS_ICONS.get(w['predicted_class'], '')} {w['predicted_class']}" for w in timeline_data]

    fig, ax = plt.subplots(figsize=(max(8, len(times) * 1.5), 3.2))
    bars = ax.bar(times, confs, color="#4299e1", edgecolor="#2b6cb0", width=0.6)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Window Confidence (%)", fontsize=9, fontweight="bold", color="#2d3748")
    ax.set_xlabel("Time Segment", fontsize=9, fontweight="bold", color="#2d3748")
    ax.set_title("Temporal Sliding Window Classification", fontsize=10, fontweight="bold", color="#1a202c")

    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 2.0,
            f"{labels[i]}\n({h:.0f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#1a365d",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    return fig
