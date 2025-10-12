import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .metrics import CoralMetrics


class CoralTrainer:
    """Trainer for coral health classification model."""

    def __init__(self, model, device='cuda', class_names=None):
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.class_names = class_names or ["Healthy", "Unhealthy", "Dead"]
        self.num_classes = len(self.class_names)

    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        metrics = CoralMetrics(num_classes=self.num_classes, class_names=self.class_names)

        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            pbar.set_postfix(loss=f"{running / max(1, pbar.n):.4f}")

            avg_train = running / max(1, len(train_loader))
            history["train_loss"].append(avg_train)

            # Validation
            if val_loader is not None:
                val_loss, val_metrics = self.evaluate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                CoralMetrics.print_metrics(val_metrics)

            if scheduler is not None:
                scheduler.step(val_metrics['loss'])

            # Save best model
            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                torch.save(self.model.state_dict(), 'resnet/best_model.pth')
                print(f"✓ Best model saved with F1: {best_val_f1:.4f}")

        # Print final validation metrics
        print("\n" + "="*60)
        print("FINAL VALIDATION METRICS")
        print("="*60)
        metric_printer = CoralMetrics(num_classes=self.num_classes, class_names=self.class_names)
        metric_printer.print_metrics(val_metrics)

        return metrics_history
