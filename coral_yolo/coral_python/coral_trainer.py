# trainers/coral_trainer.py
"""Custom trainer subclass for coral bleaching detection with YOLOv11."""

from ultralytics.engine.trainer import BaseTrainer
import torch
import torch.optim as optim
from coral_python.losses import CoralDetectionLoss
from coral_python.metrics import CoralMetrics

class CoralTrainer(BaseTrainer):
    """Extends YOLO's BaseTrainer with custom loss, optimizer, and metrics."""

    def __init__(self, *args, **kwargs):
        """Initialize trainer and force device to CPU."""
        super().__init__(*args, **kwargs)
        self.device = torch.device("cpu")  # Force CPU usage always

    def build_loss(self, model):
        """Builds the custom coral loss function."""
        return CoralDetectionLoss(model)

    def build_optimizer(self, model, hyp):
        """Builds a custom optimizer (AdamW with YOLO hyperparams)."""
        return optim.AdamW(
            model.parameters(),
            lr=hyp.get("lr0", 1e-3),
            betas=(hyp.get("momentum", 0.937), 0.999),
            weight_decay=hyp.get("weight_decay", 5e-4)
        )

    def build_metrics(self, args):
        """Builds custom coral metrics for validation."""
        return CoralMetrics()

    def custom_epoch_end(self, metrics):
        """Optional: logs extra metrics after each epoch."""
        print(f"Epoch finished. Coral metrics: {metrics}")
