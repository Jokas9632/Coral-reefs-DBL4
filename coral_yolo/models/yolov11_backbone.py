#YOLOv11 Backbone
import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOv11Backbone(nn.Module):
    #higher weights backbone
    def __init__(self, weights: str = "yolo11s.pt", out_channels: int = 512, freeze: bool = True):
        super().__init__()

        #Load YOLO backbone
        yolo_model = YOLO(weights).model
        self.backbone = yolo_model.model[0]  

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 640, 640)
            y = self.backbone(dummy)
            if isinstance(y, list):
                y = y[-1]
            self.out_channels = y.shape[1]

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
