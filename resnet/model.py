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
        # Make sure ECA params live in fp32 (safe/stable)
        self.conv.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) — could be fp16 under autocast
        in_dtype = x.dtype

        # descriptors
        y = self.avg_pool(x)                # (B, C, 1, 1) — same dtype as x
        y = y.squeeze(-1).transpose(1, 2)   # (B, 1, C)

        # Force the small Conv1d + sigmoid to run in fp32, then cast back
        with torch.cuda.amp.autocast(enabled=False):
            y32 = y.float()                 # (B, 1, C) -> fp32
            a32 = self.conv(y32)            # fp32 conv1d
            a32 = self.sigmoid(a32)         # fp32 sigmoid

        a = a32.to(in_dtype).transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1), back to x dtype
        return x * a

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
        self._remove_eca_hooks()

    def _make_hook(conv3: nn.Conv2d):
        eca = ECALayer(conv3.out_channels, k_size=self.eca_k_size)
        # Put ECA alongside conv3 (same device); keep its weights in fp32
        eca.to(device=conv3.weight.device, dtype=torch.float32)

        def hook_fn(module, inputs, output):
            # output is (B, C, H, W) after conv3; it may be fp16 under AMP
            out_dtype = output.dtype
            with torch.cuda.amp.autocast(enabled=False):
                out32 = output.float()
                eca_out32 = eca(out32)      # ECA forward already uses fp32 internally
            return eca_out32.to(out_dtype)  # hand back same dtype that Bottleneck expects

        return hook_fn

    from torchvision.models.resnet import Bottleneck
    for m in self.resnet.modules():
        if isinstance(m, Bottleneck) and isinstance(m.conv3, nn.Conv2d):
            handle = m.conv3.register_forward_hook(_make_hook(m.conv3))
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
