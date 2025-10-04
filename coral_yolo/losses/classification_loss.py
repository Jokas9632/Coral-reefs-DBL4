"""Cross-entropy loss for 2-class bleaching classification."""
import torch.nn as nn

class CoralClassificationLoss(nn.Module):
    """Applies CrossEntropyLoss over 2-logit outputs."""
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, labels):
        """Computes CE loss for (B,2) logits vs (B,) int labels."""
        return self.ce(logits, labels)
