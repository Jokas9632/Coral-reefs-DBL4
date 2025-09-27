# metrics/coral_metrics.py
"""Defines evaluation metrics tailored to coral bleaching detection."""

class CoralMetrics:
    """Tracks precision, recall, and F1 for coral detection."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Resets counters for a fresh evaluation run."""
        self.tp = self.fp = self.fn = 0

    def update(self, preds, gts):
        """Updates counters based on predictions vs ground truth."""
        # stub: implement box matching + TP/FP/FN counting
        pass

    def compute(self):
        """Computes final precision, recall, and F1 scores."""
        precision = self.tp / (self.tp + self.fp + 1e-6)
        recall = self.tp / (self.tp + self.fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return {"precision": precision, "recall": recall, "f1": f1}
