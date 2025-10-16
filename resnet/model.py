# model.py
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import Bottleneck


# =========================
# Efficient Channel Attention (AMP-safe)
# =========================
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
        self.conv.float()  # keep ECA weights in fp32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        in_dtype = x.dtype
        y = self.avg_pool(x).squeeze(-1).transpose(1, 2)  # (B, 1, C)

        y32 = y.float()
        a32 = self.conv(y32)
        a32 = self.sigmoid(a32)

        a = a32.to(in_dtype).transpose(1, 2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * a


# =========================
# Adaptive Concat Pool (avg + max) with reduction back to C
# =========================
class AdaptiveConcatPool2dReduce(nn.Module):
    """
    AdaptiveConcatPool2d with channel reduction back to C via 1x1 conv.
    - Takes both GAP and GMP: concat along channel dim -> (B, 2C, 1, 1)
    - Reduces back to (B, C, 1, 1) with a 1x1 conv (fp32 for stability).
    Keeps FC input size unchanged (2048 for ResNet50), so you can enable/disable at runtime.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.reduce = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.reduce.float()  # keep in fp32 for safety

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        a = self.avg(x)
        m = self.max(x)
        cat = torch.cat([a, m], dim=1)      # (B, 2C, 1, 1)
        cat32 = cat.float()
        y32 = self.reduce(cat32)            # (B, C, 1, 1) fp32
        return y32.to(in_dtype)             # cast back to input dtype


class CoralResNet(nn.Module):
    """
    ResNet50 backbone with optional ECA and optional Adaptive Concat Pooling.
    - ECA is injected via forward hooks after each Bottleneck's conv3 (pretrained-safe).
    - ConcatPool intercepts the avgpool output via a forward hook, runs (avg+max)+1x1 reduce,
      returns the same shape as avgpool ((B, C, 1, 1)), so the FC head stays the same.
    - Both features are AMP-safe and device-agnostic (moved to the runtime device).
    """
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_eca: bool = True,
        eca_k_size: int = 3,
        use_concat_pool: bool = False,
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

        # Classification head (kept at input 2048; concat-pool reduces back to 2048)
        in_feats = self.resnet.fc.in_features  # 2048 for ResNet50
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_feats, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        # Feature flags
        self.use_eca = use_eca
        self.eca_k_size = eca_k_size
        self.use_concat_pool = use_concat_pool

        # Hook handles
        self._eca_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._pool_hooks: list[torch.utils.hooks.RemovableHandle] = []

        # Attach as requested
        if self.use_eca:
            self._attach_eca_hooks()
        if self.use_concat_pool:
            self._attach_concatpool_hook()

    # ===== ECA hook plumbing =====
    def _attach_eca_hooks(self) -> None:
        self._remove_eca_hooks()

        def _make_eca_hook(conv3: nn.Conv2d):
            eca = ECALayer(conv3.out_channels, k_size=self.eca_k_size)
            eca.to(dtype=torch.float32)  # device set on-the-fly

            def hook_fn(module, inputs, output):
                # output: (B, C, H, W)
                out = output
                out_dtype = out.dtype
                out_device = out.device

                # move ECA to runtime device if needed
                if next(eca.parameters()).device != out_device:
                    eca.to(device=out_device, dtype=torch.float32)

                out32 = out.float()
                eca_out32 = eca(out32)
                return eca_out32.to(out_dtype)

            return hook_fn

        for m in self.resnet.modules():
            if isinstance(m, Bottleneck) and isinstance(m.conv3, nn.Conv2d):
                h = m.conv3.register_forward_hook(_make_eca_hook(m.conv3))
                self._eca_hooks.append(h)

    def _remove_eca_hooks(self) -> None:
        for h in self._eca_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._eca_hooks = []

    def enable_eca(self, k_size: int | None = None) -> None:
        if k_size is not None:
            self.eca_k_size = k_size
        if not self.use_eca:
            self.use_eca = True
        self._attach_eca_hooks()

    def disable_eca(self) -> None:
        self.use_eca = False
        self._remove_eca_hooks()

    # ===== Adaptive Concat Pool hook plumbing (intercepts avgpool) =====
    def _attach_concatpool_hook(self) -> None:
        self._remove_concatpool_hook()

        # Determine channels C (ResNet50 -> 2048). We’ll still infer at runtime to be safe.
        pool_module = self.resnet.avgpool  # we hook this module

        # Tiny module; we instantiate lazily on first call (need C)
        state = {"acpr": None}  # closure state to cache the reducer

        def hook_fn(module, inputs, output):
            # inputs[0] is the activation BEFORE avgpool (B, C, H, W)
            if not self.use_concat_pool:
                return output  # pass-through if disabled at call time

            x = inputs[0]
            C = x.shape[1]
            x_device = x.device
            x_dtype = x.dtype

            # Lazy-create the ACP reducer on first use, on correct device
            acpr = state["acpr"]
            if acpr is None or next(acpr.reduce.parameters()).device != x_device:
                acpr = AdaptiveConcatPool2dReduce(C).to(device=x_device, dtype=torch.float32)
                state["acpr"] = acpr

            # Run in fp32 for stability, then cast back
            x32 = x.float()
            y32 = acpr(x32)        # (B, C, 1, 1) fp32
            return y32.to(x_dtype) # keep shape identical to avgpool output

        h = pool_module.register_forward_hook(hook_fn)
        self._pool_hooks.append(h)

    def _remove_concatpool_hook(self) -> None:
        for h in self._pool_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._pool_hooks = []

    def enable_concat_pool(self) -> None:
        """Enable Adaptive Concat Pool (avg+max with reduction)."""
        self.use_concat_pool = True
        if not self._pool_hooks:
            self._attach_concatpool_hook()

    def disable_concat_pool(self) -> None:
        """Disable Adaptive Concat Pool (revert to plain avgpool)."""
        self.use_concat_pool = False
        # keep hook attached for zero-overhead pass-through, or remove if you prefer:
        # self._remove_concatpool_hook()

    # ===== standard nn.Module API =====
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

    def unfreeze_backbone(self) -> None:
        for p in self.resnet.parameters():
            p.requires_grad = True
