# losses/coral_loss.py
"""Defines custom loss functions for coral bleaching detection."""

import torch.nn as nn

class CoralDetectionLoss(nn.Module):
    """Custom coral loss (extendable for class imbalance, etc.)."""

    def __init__(self, model):
        super().__init__()
        # For now, a simple BCE; can wrap YOLO's loss if needed
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        """Computes loss given predictions and ground truths."""
        return self.bce(preds, targets)  # stub for now
