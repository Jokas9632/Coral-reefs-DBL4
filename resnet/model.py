# model.py
import torch
import torch.nn as nn
from torchvision import models

class GeM(nn.Module):
    """Generalized Mean Pooling (learnable)."""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, C)
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=(-2, -1)).pow(1.0 / self.p)
        return x

class CoralResNet(nn.Module):
    """
    ResNet-50 classifier with optional GeM pooling and a BN head.
    API is compatible with your existing pipeline:
      CoralResNet(num_classes=..., pretrained=True, freeze_backbone=True)
    Outputs logits for CrossEntropyLoss.
    """
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_gem: bool = True,         # <-- GeM enabled by default
        hidden: int = 512,            # head size
        dropout: float = 0.2,
        use_v2_weights: bool = False  # keep False to avoid changing behavior unless you want V2
    ):
        super().__init__()

        # Load ResNet-50
        if pretrained:
            if use_v2_weights:
                weights = models.ResNet50_Weights.IMAGENET1K_V2
            else:
                weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            weights = None
        self.backbone = models.resnet50(weights=weights)

        # Strip original classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Replace pooling
        if use_gem:
            # disable built-in GAP and use GeM → (B, C)
            self.backbone.avgpool = nn.Identity()
            self.pool = GeM()
            self._pool_outputs_flat = True
        else:
            # keep a standard GAP → (B, C, 1, 1) then flatten
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self._pool_outputs_flat = False

        # Classifier head (stable & light)
        self.head = nn.Sequential(
            nn.BatchNorm1d(num_features),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(num_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

        # Optionally freeze backbone (keeps head & GeM trainable)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard ResNet-50 stem + layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Global pooling
        if self._pool_outputs_flat:
            feats = self.pool(x)                       # (B, C)
        else:
            feats = self.pool(x).flatten(1)            # (B, C)

        # Classifier → logits
        logits = self.head(feats)
        return logits
