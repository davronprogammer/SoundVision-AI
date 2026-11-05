"""Tests for model architectures and parameter configurations."""

import pytest
import torch

from soundvision.models.model_factory import build_model
from soundvision.models.resnet import SoundVisionResNet


def test_model_scratch_output_shape():
    model = build_model("scratch", num_classes=10, in_channels=3)
    batch_size = 4
    x = torch.randn(batch_size, 3, 128, 216)
    out = model(x)

    assert out.shape == (batch_size, 10)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_model_feature_extraction_freeze():
    model = build_model("feature_extraction", num_classes=10)

    # Backbone parameters should have requires_grad=False
    for name, param in model.named_parameters():
        if "fc" in name:
            assert param.requires_grad is True, f"Head parameter {name} should be trainable"
        else:
            assert param.requires_grad is False, f"Backbone parameter {name} should be frozen"

    # Forward pass
    x = torch.randn(2, 3, 128, 216)
    out = model(x)
    assert out.shape == (2, 10)


def test_model_parameter_groups():
    model = build_model("fine_tuning", num_classes=10)
    groups = model.get_parameter_groups(lr_backbone=1e-4, lr_head=1e-3, weight_decay=1e-4)

    assert len(groups) == 2
    assert groups[0]["lr"] == 1e-4
    assert groups[1]["lr"] == 1e-3
    assert groups[0]["weight_decay"] == 1e-4
