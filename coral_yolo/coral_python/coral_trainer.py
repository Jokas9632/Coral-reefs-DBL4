# trainers/coral_trainer.py
"""Custom trainer subclass for coral bleaching detection with YOLOv11."""

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.torch_utils import smart_optimizer
from ..losses.coral_loss import CoralDetectionLoss
from ..metrics.coral_metrics import CoralMetrics

class CoralTrainer(BaseTrainer):
    """Extends YOLO's BaseTrainer with custom loss, optimizer, and metrics."""

    def build_loss(self, model):
        """Builds the custom coral loss function."""
        return CoralDetectionLoss(model)

    def build_optimizer(self, model, hyp):
        """Builds a custom optimizer (default = AdamW)."""
        return smart_optimizer(
            model,
            name="AdamW",
            lr=hyp['lr0'],
            momentum=hyp['momentum'],
            decay=hyp['weight_decay']
        )

    def build_metrics(self, args):
        """Builds custom coral metrics for validation."""
        return CoralMetrics()

    def custom_epoch_end(self, metrics):
        """Optional: logs extra metrics after each epoch."""
        print(f"Epoch finished. Coral metrics: {metrics}")
