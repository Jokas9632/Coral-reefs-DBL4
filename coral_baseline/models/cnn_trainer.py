import torch
import torch.nn as nn
from tqdm import tqdm
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .cnn_metrics import CoralMetrics  


class CoralCNNTrainer:
    """Trainer for the custom CNN model for coral health classification."""

    def __init__(self, model, device='cuda', class_names=None):
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.class_names = class_names or ["Healthy", "Bleached"]
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

            # Update metrics
            total_loss += loss.item()
            metrics.update(outputs.detach(), labels)

            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        epoch_metrics = metrics.compute()
        epoch_metrics['loss'] = avg_loss

        return epoch_metrics

    def validate(self, dataloader, criterion):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        metrics = CoralMetrics(num_classes=self.num_classes, class_names=self.class_names)

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Validation")
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                # Update metrics
                total_loss += loss.item()
                metrics.update(outputs, labels)

                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        val_metrics = metrics.compute()
        val_metrics['loss'] = avg_loss

        return val_metrics

    def fit(self, train_loader, val_loader, epochs, optimizer, criterion, scheduler=None):
        """
        Train the model for multiple epochs.

        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            optimizer: Optimizer (e.g., Adam, SGD)
            criterion: Loss function (e.g., CrossEntropyLoss)
            scheduler: Optional learning rate scheduler
        """
        best_val_f1 = 0.0
        metrics_history = {'train': [], 'val': []}

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)

            # Train
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            metrics_history['train'].append(train_metrics)

            print(f"\nTraining   - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1_macro']:.4f}")

            # Validate
            val_metrics = self.validate(val_loader, criterion)
            metrics_history['val'].append(val_metrics)

            print(f"Validation - Loss: {val_metrics['loss']:.4f}, "
                  f"Acc: {val_metrics['accuracy']:.4f}, "
                  f"F1: {val_metrics['f1_macro']:.4f}")

            # Learning rate scheduler step
            if scheduler is not None:
                scheduler.step(val_metrics['loss'])

            # Save best model
            if val_metrics['f1_macro'] > best_val_f1:
                best_val_f1 = val_metrics['f1_macro']
                torch.save(self.model.state_dict(), 'cnn/best_model.pth')
                print(f"✓ Best model saved with F1: {best_val_f1:.4f}")

        # Print final validation metrics
        print("\n" + "="*60)
        print("FINAL VALIDATION METRICS")
        print("="*60)
        metric_printer = CoralMetrics(num_classes=self.num_classes, class_names=self.class_names)
        metric_printer.print_metrics(val_metrics)

        return metrics_history
