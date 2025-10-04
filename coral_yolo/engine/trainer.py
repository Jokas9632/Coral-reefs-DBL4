"""Generic trainer with simple CPU/GPU switching and logging."""
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader
from coral_yolo.losses.classification_loss import CoralClassificationLoss
from coral_yolo.engine.metrics import ClsPRF1

class Trainer:
    """Runs training/validation for the coral classifier, with device auto-switch."""
    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                 device: str | None = None):
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = CoralClassificationLoss()
        self.metric = ClsPRF1(pos_class=1)

    def _run_epoch(self, loader: DataLoader, train: bool = True) -> Dict[str, Any]:
        """Runs one epoch and returns averaged loss + metrics."""
        self.model.train(mode=train)
        self.metric.reset()
        total_loss, n = 0.0, 0
        for batch in loader:
            images = batch["image"].to(self.device)    # Bx3xHxW
            masks  = batch["mask"].to(self.device)     # Bx1xHxW
            labels = batch["label"].to(self.device)    # B

            logits = self.model(images, masks)         # Bx2
            loss = self.loss_fn(logits, labels)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            self.metric.update(logits.detach(), labels.detach())
            total_loss += float(loss.item()) * labels.size(0)
            n += labels.size(0)

        stats = self.metric.compute()
        stats["loss"] = total_loss / max(1, n)
        stats["samples"] = n
        return stats

    def fit(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 50, verbose: bool = True):
        """Trains for N epochs with validation after each epoch."""
        for ep in range(1, epochs + 1):
            tr = self._run_epoch(train_loader, train=True)
            vl = self._run_epoch(val_loader, train=False)
            if verbose:
                print(f"[Epoch {ep:03d}] "
                      f"train_loss={tr['loss']:.4f}  "
                      f"val_loss={vl['loss']:.4f}  "
                      f"P={vl['precision']:.3f} R={vl['recall']:.3f} F1={vl['f1']:.3f}  "
                      f"(val_n={vl['samples']})")
