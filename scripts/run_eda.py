"""Exploratory Data Analysis script for SoundVision AI."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from soundvision.config import load_config
from soundvision.data.dataset import SELECTED_CLASSES
from soundvision.data.preprocessing import AudioPreprocessor


def run_eda(config_path: str = "configs/config.yaml") -> None:
    cfg = load_config(config_path)
    meta_path = Path(cfg.dataset.meta_csv)
    audio_dir = Path(cfg.dataset.audio_dir)
    figures_dir = Path(cfg.paths.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Running Exploratory Data Analysis (EDA)...")
    df = pd.read_csv(meta_path)
    subset_df = df[df["category"].isin(SELECTED_CLASSES)].copy()

    # 1. Plot Class Distribution by Fold
    plt.figure(figsize=(10, 5))
    fold_class_df = subset_df.groupby(["category", "fold"]).size().unstack(fill_value=0)
    fold_class_df.plot(kind="bar", stacked=True, colormap="viridis", figsize=(12, 6))
    plt.title("Sample Distribution by Category and Fold", fontsize=14, fontweight="bold")
    plt.xlabel("Sound Category", fontsize=11)
    plt.ylabel("Number of Audio Recordings", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="ESC-50 Fold")
    plt.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(figures_dir / "class_fold_distribution.png", dpi=300)
    plt.close()

    # 2. Preprocessor for Spectrogram Extraction
    preprocessor = AudioPreprocessor(
        sample_rate=cfg.preprocessing.sample_rate,
        n_fft=cfg.preprocessing.n_fft,
        hop_length=cfg.preprocessing.hop_length,
        n_mels=cfg.preprocessing.n_mels,
        target_duration=cfg.dataset.target_duration_sec,
    )

    # 3. Representative Mel-Spectrograms Gallery (1 per class)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for idx, category in enumerate(SELECTED_CLASSES):
        sample_row = subset_df[subset_df["category"] == category].iloc[0]
        audio_path = audio_dir / sample_row["filename"]

        waveform, sr = preprocessor.load_audio(audio_path)
        waveform = preprocessor.adjust_duration(waveform)
        mel_db = preprocessor.extract_mel_spectrogram(waveform).squeeze(0).numpy()

        ax = axes[idx]
        im = ax.imshow(
            mel_db,
            origin="lower",
            aspect="auto",
            cmap="magma",
            extent=[0, cfg.dataset.target_duration_sec, 0, sr // 2],
        )
        ax.set_title(f"{category.capitalize()} ({sample_row['filename']})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel("Frequency (Hz)", fontsize=9)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Log-Power (dB)")
    plt.suptitle("Representative Mel-Spectrograms (10 Environmental Classes)", fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(figures_dir / "representative_mel_spectrograms.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"EDA completed. Figures saved to '{figures_dir}'.")


if __name__ == "__main__":
    run_eda()
