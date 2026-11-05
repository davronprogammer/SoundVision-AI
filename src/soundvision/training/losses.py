"""Loss functions for SoundVision AI."""

import torch
import torch.nn as nn


def get_loss_function(label_smoothing: float = 0.0) -> nn.Module:
    """Returns CrossEntropyLoss with optional label smoothing."""
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
