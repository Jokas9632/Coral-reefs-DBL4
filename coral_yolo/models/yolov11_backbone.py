"""Wraps an Ultralytics YOLOv11 model and exposes its backbone features."""
from typing import Optional
import torch
import torch.nn as nn

try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None

class YOLOv11Backbone(nn.Module):
    """Loads a YOLOv11 model and returns the pre-head backbone as a feature extractor."""
    def __init__(self, weights: str = "yolov11s.pt", out_channels: Optional[int] = None, freeze: bool = True):
        super().__init__()
        if YOLO is None:
            raise ImportError("ultralytics is required. `pip install ultralytics`")
        self.yolo = YOLO(weights)  # loads weights & model topology
        self.backbone = self._extract_backbone(self.yolo.model)
        self.out_channels = out_channels or self._infer_out_channels(self.backbone)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False

    @staticmethod
    def _extract_backbone(model: nn.Module) -> nn.Sequential:
        """Builds a Sequential of all layers up to (but excluding) detection/seg heads."""
        # Ultralytics models often nest as model.model (ModuleList/Sequential); handle robustly.
        core = getattr(model, "model", model)
        layers = []
        for m in core.children():
            name = m.__class__.__name__.lower()
            if any(k in name for k in ("detect", "segment", "pose", "obb")):
                break
            layers.append(m)
        if not layers:
            # Fallback: try iterating one level deeper (for variants that wrap in ModuleList)
            for m in core:
                name = m.__class__.__name__.lower()
                if any(k in name for k in ("detect", "segment", "pose", "obb")):
                    break
                layers.append(m)
        if not layers:
            raise RuntimeError("Could not isolate YOLO backbone (no layers before head were found).")
        return nn.Sequential(*layers)

    @staticmethod
    def _infer_out_channels(backbone: nn.Sequential) -> int:
        """Infers the output channels by running a dummy pass (640×640) on CPU."""
        with torch.no_grad():
            x = torch.zeros(1, 3, 640, 640)
            y = backbone(x)
            return y.shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Emits backbone feature map (B, C, H, W)."""
        return self.backbone(x)
