"""ResNet-18 model architecture customized for environmental sound classification."""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class SoundVisionResNet(nn.Module):
    """Customized ResNet-18 for audio Mel-spectrogram classification.

    Supports from-scratch training, feature extraction (frozen backbone), and fine-tuning.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
        in_channels: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.dropout_rate = dropout
        self.in_channels = in_channels

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_model = resnet18(weights=weights)

        # Adapt first conv layer if in_channels != 3
        if in_channels != 3:
            old_conv = base_model.conv1
            base_model.conv1 = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            if pretrained and in_channels == 1:
                # Average weights of the 3 channels
                with torch.no_grad():
                    base_model.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        # Backbone layers (all except fc)
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        # Custom classification head
        in_features = base_model.fc.in_features  # 512 for ResNet18
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_features, num_classes)

        # Initialize head properly
        self._init_head()

        # Apply freezing if feature extraction
        if freeze_backbone:
            self.set_backbone_trainable(False)

    def _init_head(self) -> None:
        """Kaiming normal initialization for newly added head layers."""
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freezes or unfreezes all backbone feature extractor layers."""
        backbone_modules = [
            self.conv1,
            self.bn1,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
        ]
        for module in backbone_modules:
            for param in module.parameters():
                param.requires_grad = trainable

    def train(self, mode: bool = True):
        """Override train to keep BatchNorm in eval mode if backbone is frozen.

        This prevents exploding gradients / corrupted running stats during feature extraction.
        """
        super().train(mode)
        if self.freeze_backbone and mode:
            # Keep backbone batchnorms in eval mode
            for m in [self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
                for submodule in m.modules():
                    if isinstance(submodule, nn.BatchNorm2d):
                        submodule.eval()
        return self

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass up to global average pooling."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass returning unnormalized class logits. Shape: (batch_size, num_classes)."""
        feats = self.extract_features(x)
        logits = self.fc(self.dropout(feats))
        return logits

    def get_parameter_groups(
        self, lr_backbone: float, lr_head: float, weight_decay: float = 1e-4
    ) -> List[Dict[str, object]]:
        """Returns discriminative parameter groups for optimizers (e.g. AdamW)."""
        head_params = list(self.fc.parameters())
        head_ids = set(id(p) for p in head_params)

        backbone_params = [p for p in self.parameters() if id(p) not in head_ids and p.requires_grad]

        groups = []
        if backbone_params:
            groups.append({
                "params": backbone_params,
                "lr": lr_backbone,
                "weight_decay": weight_decay,
            })
        if head_params:
            groups.append({
                "params": head_params,
                "lr": lr_head,
                "weight_decay": weight_decay,
            })
        return groups
