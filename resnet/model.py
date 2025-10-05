import torch
import torch.nn as nn
from torchvision import models


class CoralResNet(nn.Module):
    """
    ResNet-based model for coral health classification.

    Classes:
        0: Healthy
        1: Unhealthy/Bleached
        2: Dead
    """

    def __init__(self, num_classes=3, pretrained=True, freeze_backbone=False):
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
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

    def unfreeze_backbone(self):
        """Unfreeze the backbone for fine-tuning."""
        for param in self.resnet.parameters():
            param.requires_grad = True
