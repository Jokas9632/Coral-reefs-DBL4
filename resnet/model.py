import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import Bottleneck  # for type checks


class ECALayer(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), possibly fp16 under autocast
        dtype = x.dtype

        # global avg pool (keeps dtype)
        y = self.avg_pool(x)                # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(1, 2)   # (B, 1, C)

        # Do the ECA conv + sigmoid in FP32 for stability, then cast back
        # This avoids "Half input vs Float weight" runtime errors.
        with torch.cuda.amp.autocast(enabled=False):
            y32 = y.float()                 # (B, 1, C) -> fp32
            a32 = self.conv(y32)            # fp32 conv
            a32 = self.sigmoid(a32)         # fp32 sigmoid

        a = a32.to(dtype).transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1), back to original dtype
        return x * a.expand_as(x)

    """
    Efficient Channel Attention (ECA)
    Paper: ECA-Net: Efficient Channel Attention for Deep CNN (CVPR 2020)
    Uses local cross-channel interaction via 1D conv over channel-wise descriptors.
    """
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        # global average pooling -> (B, C, 1, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 1D conv over the channel dimension (C, 1) with padding to keep length
        self.conv = nn.Conv1d(
            in_channels=1, out_channels=1, kernel_size=k_size,
            padding=(k_size // 2), bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        y = self.avg_pool(x)                # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(1, 2)   # (B, 1, C)
        y = self.conv(y)                    # (B, 1, C)
        y = self.sigmoid(y).transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y.expand_as(x)


class CoralResNet(nn.Module):
    """
    ResNet-based model for coral health classification.

    Classes:
        0: Healthy
        1: Unhealthy/Bleached
        2: Dead
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_eca: bool = True,
        eca_k_size: int = 3
    ):
        super(CoralResNet, self).__init__()

        # Load pretrained ResNet50
        if pretrained:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet50(weights=None)

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Replace final fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        # === ECA integration (non-invasive; keeps pretrained weights loadable) ===
        self.use_eca = use_eca
        self.eca_k_size = eca_k_size
        self._eca_hooks = []
        if self.use_eca:
            self._attach_eca_hooks()

    def _attach_eca_hooks(self):
        """
        Attach ECA after each Bottleneck's final conv (conv3) using forward hooks.
        """
        # Clean up existing hooks if any
        self._remove_eca_hooks()

        def _make_hook(channels):
            eca = ECALayer(channels, k_size=self.eca_k_size).to(next(self.parameters()).device)

            def hook_fn(module, inputs, output):
                # output is the feature map after conv3 (before residual add+relu in Bottleneck forward)
                return eca(output)

            return hook_fn

        # Register hooks on every Bottleneck's conv3
        for m in self.resnet.modules():
            if isinstance(m, Bottleneck):
                conv3 = m.conv3
                if isinstance(conv3, nn.Conv2d):
                    handle = conv3.register_forward_hook(_make_hook(conv3.out_channels))
                    self._eca_hooks.append(handle)

    def _remove_eca_hooks(self):
        for h in self._eca_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._eca_hooks = []

    def enable_eca(self, k_size: int = None):
        """Enable (or reconfigure) ECA at runtime."""
        if k_size is not None:
            self.eca_k_size = k_size
        self.use_eca = True
        self._attach_eca_hooks()

    def disable_eca(self):
        """Disable ECA at runtime."""
        self.use_eca = False
        self._remove_eca_hooks()

    def forward(self, x):
        return self.resnet(x)

    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning."""
        for param in self.resnet.parameters():
            param.requires_grad = True
