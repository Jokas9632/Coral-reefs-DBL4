#Model Backbone 
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import Bottleneck


#ECA Layer used in Experimentation
class ECALayer(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv.float() 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        y = self.avg_pool(x).squeeze(-1).transpose(1, 2)  

        y32 = y.float()
        a32 = self.conv(y32)
        a32 = self.sigmoid(a32)

        a = a32.to(in_dtype).transpose(1, 2).unsqueeze(-1)  
        return x * a

#AdapativeConcatinationPooling used in Experimentation 
class AdaptiveConcatPool2dReduce(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.reduce = nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False)
        self.reduce.float() 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        a = self.avg(x)
        m = self.max(x)
        cat = torch.cat([a, m], dim=1)      
        cat32 = cat.float()
        y32 = self.reduce(cat32)            
        return y32.to(in_dtype)

#GeM Pooling Layer used in Experimentation               
class GeMPool2d(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True):
        super().__init__()
        self.eps = eps
        if learnable:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.tensor([p], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(min=self.eps)
        p = self.p.clamp(min=1e-3) 
        x = x.pow(p.view(1, 1, 1, 1))
        x = x.mean(dim=(-1, -2), keepdim=True)
        x = x.pow(1.0 / p.view(1, 1, 1, 1))
        return x
    
#Resnet Backbone 
class CoralResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_eca: bool = False,
        use_gem: bool = False,
        gem_p_init: float = 3.0,
        gem_learnable: bool = True,
        gem_eps: float = 1e-6,
        eca_k_size: int = 3,
        use_concat_pool: bool = False,
    ):
        super().__init__()

        if pretrained:
            self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.resnet = models.resnet50(weights=None)

        if freeze_backbone:
            for p in self.resnet.parameters():
                p.requires_grad = False

        #Classification head
        in_feats = self.resnet.fc.in_features  # 2048 for ResNet50
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_feats, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        #Attach Architecture Changes on Instance
        self.use_eca = use_eca
        self.eca_k_size = eca_k_size
        self.use_concat_pool = use_concat_pool
        self.use_gem = use_gem
        self.gem_p_init = gem_p_init
        self.gem_learnable = gem_learnable
        self.gem_eps = gem_eps
        
        self._eca_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._pool_hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._gem_hooks: list[torch.utils.hooks.RemovableHandle] = []

        if self.use_eca:
            self._attach_eca_hooks()
        if self.use_concat_pool and self.use_gem:
            #defaults to concat if both enabled
            self.use_gem = False
        if self.use_concat_pool:
            self._attach_concatpool_hook()
        if self.use_gem:
            self._attach_gem_pool_hook()

    #ECA Hook
    def _attach_eca_hooks(self) -> None:
        self._remove_eca_hooks()

        def _make_eca_hook(conv3: nn.Conv2d):
            eca = ECALayer(conv3.out_channels, k_size=self.eca_k_size)
            eca.to(dtype=torch.float32)  

            def hook_fn(module, inputs, output):
                out = output
                out_dtype = out.dtype
                out_device = out.device

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

    #Adapative Concat Pool Hook
    def _attach_concatpool_hook(self) -> None:
        self._remove_concatpool_hook()

        #Deterimine the pooling module to hook
        pool_module = self.resnet.avgpool  
        state = {"acpr": None} 

        def hook_fn(module, inputs, output):
            if not self.use_concat_pool:
                return output 

            x = inputs[0]
            C = x.shape[1]
            x_device = x.device
            x_dtype = x.dtype

            acpr = state["acpr"]
            if acpr is None or next(acpr.reduce.parameters()).device != x_device:
                acpr = AdaptiveConcatPool2dReduce(C).to(device=x_device, dtype=torch.float32)
                state["acpr"] = acpr
            x32 = x.float()
            y32 = acpr(x32)        
            return y32.to(x_dtype) 

        h = pool_module.register_forward_hook(hook_fn)
        self._pool_hooks.append(h)

    def _remove_concatpool_hook(self) -> None:
        for h in self._pool_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._pool_hooks = []

    #Live concat pool enabler - primarily used for debugging
    def enable_concat_pool(self) -> None:
        self.use_concat_pool = True
        if not self._pool_hooks:
            self._attach_concatpool_hook()

    def disable_concat_pool(self) -> None:
        self.use_concat_pool = False

    #GeM Pool Hook
    def _attach_gem_pool_hook(self) -> None:
        self._remove_gem_pool_hook()

        pool_module = self.resnet.avgpool
        state = {"gem": None}

        def hook_fn(module, inputs, output):
            if not self.use_gem or self.use_concat_pool:
                return output

            x = inputs[0]  
            x_device = x.device
            x_dtype = x.dtype

            gem: GeMPool2d | None = state["gem"]
            if (
                gem is None
                or (hasattr(gem, "p") and isinstance(gem.p, torch.Tensor) and gem.p.device != x_device)
            ):
                gem = GeMPool2d(
                    p=self.gem_p_init,
                    eps=self.gem_eps,
                    learnable=self.gem_learnable,
                ).to(device=x_device, dtype=torch.float32)
                state["gem"] = gem

            x32 = x.float()
            y32 = gem(x32)         
            return y32.to(x_dtype)

        h = pool_module.register_forward_hook(hook_fn)
        self._gem_hooks.append(h)

    def _remove_gem_pool_hook(self) -> None:
        for h in self._gem_hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._gem_hooks = []

    def enable_gem(
        self,
        p: float | None = None,
        learnable: bool | None = None,
        eps: float | None = None,
    ) -> None:
        if p is not None:
            self.gem_p_init = float(p)
        if learnable is not None:
            self.gem_learnable = bool(learnable)
        if eps is not None:
            self.gem_eps = float(eps)

        if self.use_concat_pool:
            self.disable_concat_pool()

        self.use_gem = True
        if not self._gem_hooks:
            self._attach_gem_pool_hook()

    def disable_gem(self) -> None:
        self.use_gem = False
        self._remove_gem_pool_hook()

    #Neural Net Forward Pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resnet(x)

    def unfreeze_backbone(self) -> None:
        for p in self.resnet.parameters():
            p.requires_grad = True
