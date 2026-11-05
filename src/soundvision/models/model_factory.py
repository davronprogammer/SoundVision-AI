"""Factory functions to instantiate SoundVision models for all three experiment paradigms."""

from typing import Optional
from soundvision.config import ModelConfig
from soundvision.models.resnet import SoundVisionResNet


def build_model(
    experiment_type: str,
    num_classes: int = 10,
    dropout: float = 0.3,
    in_channels: int = 3,
) -> SoundVisionResNet:
    """Builds a SoundVision model according to experiment paradigm:

    - 'scratch': ResNet-18 initialized from scratch (random weights), trainable backbone.
    - 'feature_extraction': ResNet-18 with ImageNet weights, frozen backbone, trainable head.
    - 'fine_tuning': ResNet-18 with ImageNet weights, fully trainable with discriminative LR.
    """
    exp_clean = experiment_type.strip().lower()

    if exp_clean in ["scratch", "from_scratch", "baseline_scratch"]:
        return SoundVisionResNet(
            num_classes=num_classes,
            pretrained=False,
            freeze_backbone=False,
            dropout=dropout,
            in_channels=in_channels,
        )
    elif exp_clean in ["feature_extraction", "frozen", "baseline_feature_extraction"]:
        return SoundVisionResNet(
            num_classes=num_classes,
            pretrained=True,
            freeze_backbone=True,
            dropout=dropout,
            in_channels=in_channels,
        )
    elif exp_clean in ["fine_tuning", "finetune", "baseline_fine_tuning", "improved_fine_tuning"]:
        return SoundVisionResNet(
            num_classes=num_classes,
            pretrained=True,
            freeze_backbone=False,
            dropout=dropout,
            in_channels=in_channels,
        )
    else:
        raise ValueError(
            f"Unknown experiment_type '{experiment_type}'. "
            "Supported: 'scratch', 'feature_extraction', 'fine_tuning'"
        )
