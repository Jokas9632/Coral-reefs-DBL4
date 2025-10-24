#Coral Classifier 
import torch
import torch.nn as nn
from .yolov11_backbone import YOLOv11Backbone
from .mask_encoder import MaskEncoder

class CoralClassifier(nn.Module):
    def __init__(self, yolo_weights: str = "yolo11n.pt", num_classes: int = 2, #smallest yolo model
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
        #Forward pass
        feat = self.backbone(image)                               # BxCxhxw
        mfeat = self.mask_enc(mask, feat_hw=(feat.shape[2], feat.shape[3]))  # BxCm×hxw
        fused = torch.cat([feat, mfeat], dim=1)                   # Bx(C+Cm)×hxw
        fused = self.fuse(fused)                                  # BxC×hxw
        pooled = self.pool(fused).flatten(1)                      # BxC
        logits = self.head(pooled)                                # BxK
        return logits
    
    def rgb_mask_to_binary(mask_batch: torch.Tensor) -> torch.Tensor:
        #Convert RGB mask to binary coral mask
        coral_mask = (mask_batch.sum(dim=1, keepdim=True) > 0).float()
        return coral_mask

    

