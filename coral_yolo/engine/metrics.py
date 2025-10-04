"""Lightweight classification metrics."""
import torch

class ClsPRF1:
    """Accumulates TP/FP/FN for binary classification and computes P/R/F1."""
    def __init__(self, pos_class: int = 1):
        self.pos = pos_class
        self.reset()

    def reset(self):
        """Resets counters."""
        self.tp = self.fp = self.fn = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, thr: float = 0.5):
        """Updates counts given logits and labels."""
        probs = torch.softmax(logits, dim=1)[:, self.pos]
        preds = (probs >= thr).long()
        self.tp += int(((preds == 1) & (labels == 1)).sum())
        self.fp += int(((preds == 1) & (labels == 0)).sum())
        self.fn += int(((preds == 0) & (labels == 1)).sum())

    def compute(self) -> dict:
        """Computes precision, recall, and F1."""
        precision = self.tp / max(1, (self.tp + self.fp))
        recall    = self.tp / max(1, (self.tp + self.fn))
        f1        = 2 * precision * recall / max(1e-6, (precision + recall))
        return {"precision": precision, "recall": recall, "f1": f1}
