# model.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import Bottleneck


class ECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA) — AMP-safe, device-agnostic.
    We compute the tiny Conv1d + sigmoid in fp32 for stability, then cast
    the attention back to the input dtype and scale the features.
    """
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # keep ECA weights in fp32 for numeric stability
        self.conv.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        in_dtype = x.dtype
        # Global average pool -> channel descriptor (B, 1, C)
        y = self.avg_pool(x).squeeze(-1).transpose(1, 2)  # (B, 1, C)

        # Do attention strictly in fp32 (no autocast context here)
        y32 = y.float()
        a32 = self.conv(y32)         # (B, 1, C) in fp32
        a32 = self.sigmoid(a32)      # (B, 1, C) in fp32

        # Shape back to (B, C, 1, 1) and cast to input dtype
        a = a32.to(in_dtype).transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * a


class CoralResNet(nn.Module):
    """
    ResNet50 backbone with optional ECA after each Bottleneck's conv3.
    - Hooks add ECA without altering module names => pretrained weights stay compatible.
    - ECA is AMP-safe and moves to the correct device at runtime.
    """
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_eca: bool = True,
        eca_k_size: int = 3,
    ):
        super().__init__()

        # Backbone
        if pretrained:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet50(weights=None)

        # Optional: freeze backbone
        if freeze_backbone:
            for p in self.resnet.parameters():
                p.requires_grad = False

        # Classification head
        in_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_feats, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        # ECA config
        self.use_eca = use_eca
        self.eca_k_size = eca_k_size
        self._eca_hooks: list[torch.utils.hooks.RemovableHandle] = []

        if self.use_eca:
            self._attach_eca_hooks()

    # ===== ECA hook plumbing =====
    def _attach_eca_hooks(self) -> None:
        """Attach ECA after each Bottleneck's conv3 in a device/dtype-safe way."""
        self._remove_eca_hooks()

        def _make_hook(conv3: nn.Conv2d):
            # Create ECA in fp32; we'll move it to the correct device at runtime
            eca = ECALayer(conv3.out_channels, k_size=self.eca_k_size)
            eca.to(dtype=torch.float32)  # device set on-the-fly

            def hook_fn(module, inputs, output):
                # output: (B, C, H, W) — could be CPU or CUDA, fp16 or fp32
                out = output
                out_dtype = out.dtype
                out_device = out.device

                # Ensure ECA lives on the same device as the activation
                if next(eca.parameters()).device != out_device:
                    eca.to(device=out_device, dtype=torch.float32)

                # Run ECA math in fp32, then cast back to the original dtype
                out32 = out.float()
                eca_out32 = eca(out32)
                return eca_out32.to(out_dtype)

            return hook_fn

        for m in self.resnet.modules():
            if isinstance(m, Bottleneck) and isinstance(m.conv3, nn.Conv2d):
                h = m.conv3.register_forward_hook(_make_hook(m.conv3))
                self._eca_hooks.append(h)

    def _remove_eca_hooks(self) -> None:
        for h in self._eca_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._eca_hooks = []

    def enable_eca(self, k_size: int | None = None) -> None:
        """Enable (or reconfigure) ECA at runtime."""
        if k_size is not None:
            self.eca_k_size = k_size
        if not self.use_eca:
            self.use_eca = True
        self._attach_eca_hooks()

    def disable_eca(self) -> None:
        """Disable ECA at runtime."""
        self.use_eca = False
        self._remove_eca_hooks()

    # ===== standard nn.Module API =====
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

    def unfreeze_backbone(self) -> None:
        for p in self.resnet.parameters():
            p.requires_grad = True
