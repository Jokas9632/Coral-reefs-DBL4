"""End model: YOLOv11 backbone + mask encoder + fusion + classification head."""
import torch
import torch.nn as nn
from .yolov11_backbone import YOLOv11Backbone
from .mask_encoder import MaskEncoder

class CoralClassifier(nn.Module):
    """Fuses image features with mask features and predicts bleaching class."""
    def __init__(self, yolo_weights: str = "yolov11s.pt", num_classes: int = 2,
                 mask_channels: int = 64, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = YOLOv11Backbone(weights=yolo_weights, freeze=freeze_backbone)
        self.mask_enc = MaskEncoder(out_channels=mask_channels)
        c_back = self.backbone.out_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(c_back + mask_channels, c_back, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c_back),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(c_back, num_classes)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Runs forward pass and returns logits [B, num_classes]."""
        feat = self.backbone(image)                               # BxCxhxw
        mfeat = self.mask_enc(mask, feat_hw=(feat.shape[2], feat.shape[3]))  # BxCm×hxw
        fused = torch.cat([feat, mfeat], dim=1)                   # Bx(C+Cm)×hxw
        fused = self.fuse(fused)                                  # BxC×hxw
        pooled = self.pool(fused).flatten(1)                      # BxC
        logits = self.head(pooled)                                # BxK
        return logits
