"""UI styling and visualization helpers for SoundVision AI Streamlit app."""

import io
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import streamlit as st
import torch

from soundvision.data.preprocessing import AudioPreprocessor

CUSTOM_CSS = """
<style>
    /* Clean, professional styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a2a3a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2b6cb0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #718096;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .prediction-box {
        background: linear-gradient(135deg, #ebf8ff 0%, #edf2f7 100%);
        border: 1px solid #bee3f8;
        border-radius: 10px;
        padding: 1.25rem;
        margin: 1rem 0;
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
</style>
"""


def apply_custom_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def plot_waveform_and_spectrogram(
    waveform: np.ndarray,
    sr: int,
    mel_db: np.ndarray,
) -> plt.Figure:
    """Creates a side-by-side or stacked waveform and Mel-spectrogram figure."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 4.5), sharex=False)

    # Waveform
    time_axis = np.linspace(0, len(waveform) / sr, num=len(waveform))
    ax1.plot(time_axis, waveform, color="#2b6cb0", lw=0.8)
    ax1.set_title("Audio Waveform", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Amplitude", fontsize=9)
    ax1.set_xlim(0, max(time_axis[-1], 0.1))
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Mel-spectrogram
    im = ax2.imshow(
        mel_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, max(time_axis[-1], 0.1), 0, sr // 2],
    )
    ax2.set_title("Log-Mel Spectrogram (dB)", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Time (s)", fontsize=9)
    ax2.set_ylabel("Frequency (Hz)", fontsize=9)

    plt.tight_layout()
    return fig
