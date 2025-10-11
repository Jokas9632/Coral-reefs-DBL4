import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def _logit_offset_from_prior(p: float) -> float:
    """Return log-odds log(p/(1-p)) with numeric safety."""
    p = float(max(min(p, 1.0 - 1e-8), 1e-8))
    return math.log(p / (1.0 - p))


class DualPool(nn.Module):
    """Concatenate Global Average Pool and Global Max Pool -> (N, 2C)."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)
        gmp = F.adaptive_max_pool2d(x, 1).flatten(1)
        return torch.cat([gap, gmp], dim=1)


class CoralResNet(nn.Module):
    """
    Binary classifier with a ResNet-50 backbone (unchanged) + DualPool head.
    Changes included:
      1) Single-logit head
      2) Prior-aware bias init (requires pos_prior)
      3) Logit adjustment offset (tau * logit_prior) in forward
      4) DualPool (GAP ⊕ GMP) features
      5) Keep everything else minimal and compatible
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        pos_prior: float = 0.5,     # set to your train prior, e.g. 0.08
        tau: float = 0.0,           # 0.0 disables logit adjustment; 1.0 is a good starting point
        dropout: float = 0.2
    ):
        super().__init__()
        # --- Backbone (unchanged) ---
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)
        # keep only the conv trunk; we'll add our own head
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        if freeze_backbone:
            for p in self.stem.parameters():
                p.requires_grad = False

        # --- Head: DualPool -> BN -> Dropout -> Linear(2C -> 1) ---
        self.pool = DualPool()
        feat_dim = 2048 * 2
        self.bn = nn.BatchNorm1d(feat_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(feat_dim, 1, bias=True)   # single logit (binary)

        # (1) + (2) prior-aware bias init
        self._prior = float(pos_prior)
        with torch.no_grad():
            self.fc.bias.fill_(_logit_offset_from_prior(self._prior))

        # (3) logit adjustment strength
        # forward() will add tau * log_odds(p) to logits
        self.tau = float(tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)                       # (N, 2048, H/32, W/32)
        feats = self.pool(x)                   # (N, 4096)
        feats = self.bn(feats)
        feats = self.drop(feats)
        logits = self.fc(feats)                # (N, 1)

        # (3) optional logit adjustment (architectural shift toward positive class)
        if self.tau != 0.0:
            logits = logits + _logit_offset_from_prior(self._prior) * self.tau

        return logits  # raw logits for BCEWithLogitsLoss

    def unfreeze_backbone(self):
        for p in self.stem.parameters():
            p.requires_grad = True
