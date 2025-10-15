# model.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import Bottleneck


class ECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA) — AMP safe.
    We run the tiny Conv1d + sigmoid in fp32 and cast back to the input dtype.
    """
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        # keep ECA weights in fp32 for stability
        self.conv.float()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        # (B, C, 1, 1) -> (B, 1, C)
        y = self.avg_pool(x).squeeze(-1).transpose(1, 2)

        # do attention in fp32 regardless of outer autocast
        with torch.cuda.amp.autocast(enabled=False):
            y32 = y.float()
            a32 = self.conv(y32)
            a32 = self.sigmoid(a32)

        a = a32.to(in_dtype).transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * a


class CoralResNet(nn.Module):
    """
    ResNet-based model with optional ECA after each Bottleneck's conv3.
    Keeps torchvision state_dict compatibility (hooks, not module surgery).
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

        # backbone
        if pretrained:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet50(weights=None)

        # optional freeze
        if freeze_backbone:
            for p in self.resnet.parameters():
                p.requires_grad = False

        # head
        in_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_feats, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        # ECA
        self.use_eca = use_eca
        self.eca_k_size = eca_k_size
        self._eca_hooks = []
        if self.use_eca:
            self._attach_eca_hooks()

    # ===== ECA hook plumbing =====
    def _attach_eca_hooks(self):
        self._remove_eca_hooks()

        def _make_hook(conv3: nn.Conv2d):
            # ECA alongside conv3 on same device; keep ECA params in fp32
            eca = ECALayer(conv3.out_channels, k_size=self.eca_k_size)
            eca.to(device=conv3.weight.device, dtype=torch.float32)

            def hook_fn(module, inputs, output):
                # output is (B, C, H, W); might be fp16 under AMP
                out_dtype = output.dtype
                with torch.cuda.amp.autocast(enabled=False):
                    out32 = output.float()
                    eca_out32 = eca(out32)
                return eca_out32.to(out_dtype)

            return hook_fn

        for m in self.resnet.modules():
            if isinstance(m, Bottleneck) and isinstance(m.conv3, nn.Conv2d):
                h = m.conv3.register_forward_hook(_make_hook(m.conv3))
                self._eca_hooks.append(h)

    def _remove_eca_hooks(self):
        for h in self._eca_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._eca_hooks = []

    def enable_eca(self, k_size: int | None = None):
        if k_size is not None:
            self.eca_k_size = k_size
        self.use_eca = True
        self._attach_eca_hooks()

    def disable_eca(self):
        self.use_eca = False
        self._remove_eca_hooks()

    # ===== standard nn.Module API =====
    def forward(self, x):
        return self.resnet(x)

    def unfreeze_backbone(self):
        for p in self.resnet.parameters():
            p.requires_grad = True
