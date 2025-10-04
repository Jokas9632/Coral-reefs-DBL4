"""Small CNN that encodes a binary coral mask to match backbone feature scale."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskEncoder(nn.Module):
    """Encodes 1×HxW binary mask into C_m×h×w feature map aligned with backbone."""
    def __init__(self, out_channels: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, mask: torch.Tensor, feat_hw: tuple[int, int]) -> torch.Tensor:
        """Downscales mask to (h,w) using nearest and runs the CNN."""
        m = F.interpolate(mask, size=feat_hw, mode="nearest")
        return self.conv(m)
