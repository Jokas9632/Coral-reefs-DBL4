#Encodes  binary mask to features 
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskEncoder(nn.Module):
    #Specific definition of mask encoder
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
        #Mask downscaling on forward pass
        m = F.interpolate(mask, size=feat_hw, mode="nearest")
        return self.conv(m)
