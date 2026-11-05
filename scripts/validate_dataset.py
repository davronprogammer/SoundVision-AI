"""Script to comprehensively validate audio dataset integrity, folds, and metadata."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf

from soundvision.config import load_config
from soundvision.data.dataset import SELECTED_CLASSES


def validate_dataset(config_path: str = "configs/config.yaml") -> bool:
    cfg = load_config(config_path)
    meta_path = Path(cfg.dataset.meta_csv)
    audio_dir = Path(cfg.dataset.audio_dir)

    print("=" * 60)
    print(" SOUNDVISION AI — DATASET VALIDATION")
    print("=" * 60)

    if not meta_path.is_file():
        print(f"[FAIL] Metadata file not found at: {meta_path}")
        return False

    if not audio_dir.is_dir():
        print(f"[FAIL] Audio directory not found at: {audio_dir}")
        return False

    df = pd.read_csv(meta_path)
    print(f"[OK] Read ESC-50 metadata CSV: {len(df)} total entries across all classes.")

    subset_df = df[df["category"].isin(SELECTED_CLASSES)].copy()
    print(f"[OK] Filtered subset entries: {len(subset_df)} samples for 10 classes.")

    if len(subset_df) != 400:
        print(f"[FAIL] Expected 400 recordings, got {len(subset_df)}")
        return False

    # Check class distribution
    class_counts = subset_df["category"].value_counts()
    for cls in SELECTED_CLASSES:
        cnt = class_counts.get(cls, 0)
        if cnt != 40:
            print(f"[FAIL] Class '{cls}' has {cnt} recordings (expected 40)")
            return False
    print("[OK] Class balance: exactly 40 recordings per class for all 10 classes.")

    # Check fold distribution
    fold_counts = subset_df["fold"].value_counts().sort_index()
    for fold in [1, 2, 3, 4, 5]:
        cnt = fold_counts.get(fold, 0)
        if cnt != 80:
            print(f"[FAIL] Fold {fold} has {cnt} recordings (expected 80)")
            return False
    print("[OK] Fold distribution: exactly 80 recordings per fold (Folds 1-5).")

    # Train / Val / Test split breakdown
    train_count = len(subset_df[subset_df["fold"].isin([1, 2, 3])])
    val_count = len(subset_df[subset_df["fold"] == 4])
    test_count = len(subset_df[subset_df["fold"] == 5])
    print(f"[OK] Train set (Folds 1-3): {train_count} samples (expected 240)")
    print(f"[OK] Validation set (Fold 4): {val_count} samples (expected 80)")
    print(f"[OK] Test set (Fold 5): {test_count} samples (expected 80)")

    # Audio integrity checks
    corrupted = []
    durations = []
    sample_rates = set()

    for _, row in subset_df.iterrows():
        filepath = audio_dir / row["filename"]
        if not filepath.is_file():
            corrupted.append((row["filename"], "File missing"))
            continue

        try:
            info = sf.info(str(filepath))
            sample_rates.add(info.samplerate)
            durations.append(info.duration)

            # Check if readable
            data, _ = sf.read(str(filepath))
            if np.isnan(data).any() or np.isinf(data).any():
                corrupted.append((row["filename"], "Contains NaN/Inf"))
            if np.max(np.abs(data)) < 1e-6:
                corrupted.append((row["filename"], "Silent audio"))
        except Exception as e:
            corrupted.append((row["filename"], str(e)))

    if corrupted:
        print(f"[FAIL] Corrupted files detected: {corrupted}")
        return False

    avg_dur = np.mean(durations)
    print(f"[OK] All 400 audio files are readable and valid.")
    print(f"[OK] Original sample rates: {list(sample_rates)}")
    print(f"[OK] Clip durations: mean = {avg_dur:.2f}s, min = {np.min(durations):.2f}s, max = {np.max(durations):.2f}s")
    print("=" * 60)
    print(" DATASET VALIDATION PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    success = validate_dataset()
    sys.exit(0 if success else 1)
