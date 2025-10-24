#loss function
import torch.nn as nn

class CoralClassificationLoss(nn.Module):
    #Cross-entropy loss for coral classification
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, labels):
        return self.ce(logits, labels)
