"""Tests for official ESC-50 fold partitioning and leakage prevention."""

import numpy as np
import pandas as pd
import pytest

from soundvision.config import AppConfig
from soundvision.data.dataset import SELECTED_CLASSES


def test_fold_partitioning_disjoint():
    """Verify that train, validation, and test fold sets are strictly disjoint."""
    cfg = AppConfig()
    train_folds = set(cfg.dataset.train_folds)
    val_folds = set(cfg.dataset.val_folds)
    test_folds = set(cfg.dataset.test_folds)

    # Disjointness checks
    assert train_folds.isdisjoint(val_folds), "Train and validation folds overlap!"
    assert train_folds.isdisjoint(test_folds), "Train and test folds overlap!"
    assert val_folds.isdisjoint(test_folds), "Validation and test folds overlap!"

    # Completeness check (ESC-50 has 5 folds)
    all_folds = train_folds | val_folds | test_folds
    assert all_folds == {1, 2, 3, 4, 5}, "Fold configuration must cover exactly folds 1-5"


def test_split_sizes_and_balance():
    """Verify fold size proportions (240 train, 80 val, 80 test for 10 classes)."""
    cfg = AppConfig()
    classes = cfg.dataset.classes

    # Synthetic representative metadata dataframe
    records = []
    for fold in range(1, 6):
        for cls in classes:
            for rep in range(8):  # 8 clips per class per fold = 40 total per class
                records.append({
                    "filename": f"{fold}-{cls}-{rep}.wav",
                    "fold": fold,
                    "category": cls,
                })
    meta_df = pd.DataFrame(records)

    assert len(meta_df) == 400

    train_df = meta_df[meta_df["fold"].isin(cfg.dataset.train_folds)]
    val_df = meta_df[meta_df["fold"].isin(cfg.dataset.val_folds)]
    test_df = meta_df[meta_df["fold"].isin(cfg.dataset.test_folds)]

    assert len(train_df) == 240, f"Expected 240 train samples, got {len(train_df)}"
    assert len(val_df) == 80, f"Expected 80 val samples, got {len(val_df)}"
    assert len(test_df) == 80, f"Expected 80 test samples, got {len(test_df)}"

    # Per-class distribution in each fold
    for cls in classes:
        assert len(train_df[train_df["category"] == cls]) == 24
        assert len(val_df[val_df["category"] == cls]) == 8
        assert len(test_df[test_df["category"] == cls]) == 8
