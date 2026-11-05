"""Tests for ESC50SubsetDataset and class mappings."""

import pandas as pd
import pytest
import soundfile as sf
import torch

from soundvision.data.augmentation import AudioAugmenter
from soundvision.data.dataset import (
    CLASS_TO_IDX,
    ESC50SubsetDataset,
    IDX_TO_CLASS,
    SELECTED_CLASSES,
)
from soundvision.data.preprocessing import AudioPreprocessor


def test_class_mappings():
    assert len(SELECTED_CLASSES) == 10
    assert len(CLASS_TO_IDX) == 10
    assert len(IDX_TO_CLASS) == 10

    for idx, cls_name in enumerate(SELECTED_CLASSES):
        assert CLASS_TO_IDX[cls_name] == idx
        assert IDX_TO_CLASS[idx] == cls_name


def test_dataset_sample_loading(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    # Generate two mock WAV files
    test_files = [
        ("1-100032-A-0.wav", "dog", 1),
        ("2-100038-A-14.wav", "thunderstorm", 2),
    ]
    for fn, cls, fold in test_files:
        sr = 22050
        data = np_data = torch.sin(torch.linspace(0, 100, sr * 5)).numpy()
        sf.write(str(audio_dir / fn), data, sr)

    meta_df = pd.DataFrame([
        {"filename": fn, "category": cls, "fold": fold}
        for fn, cls, fold in test_files
    ])

    preprocessor = AudioPreprocessor(sample_rate=22050, target_duration=5.0)
    dataset = ESC50SubsetDataset(
        meta_df=meta_df,
        audio_dir=audio_dir,
        preprocessor=preprocessor,
        is_train=False,
    )

    assert len(dataset) == 2

    # Test item 0
    feature, label = dataset[0]
    assert feature.shape == (3, 128, 216)
    assert label == CLASS_TO_IDX["dog"]

    # Test item 1
    feature2, label2 = dataset[1]
    assert feature2.shape == (3, 128, 216)
    assert label2 == CLASS_TO_IDX["thunderstorm"]


def test_augmentation_specaugment():
    augmenter = AudioAugmenter(enabled=True, time_mask_max_len=10, freq_mask_max_len=10)
    spec = torch.ones(3, 128, 216)

    aug_spec = augmenter.augment_spectrogram(spec)
    assert aug_spec.shape == spec.shape
    # Masking replaces values with 0, so sum should decrease
    assert aug_spec.sum().item() < spec.sum().item()
