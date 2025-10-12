import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


class CoralMetrics:
    """Calculate classification metrics for coral health prediction."""

    def __init__(self, num_classes=3, class_names=None):
        self.num_classes = num_classes
        self.class_names = class_names or ["Healthy", "Unhealthy", "Dead"]
        self.reset()

    def reset(self):
        """Reset all metrics."""
        self.all_preds = []
        self.all_labels = []

    def update(self, preds, labels):
        """
        Update metrics with batch predictions.

        Args:
            preds: torch.Tensor of shape (batch_size, num_classes) - logits or probabilities
            labels: torch.Tensor of shape (batch_size,) - ground truth labels
        """
        if preds.dim() == 2:
            preds = torch.argmax(preds, dim=1)

        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_labels.extend(labels.cpu().numpy().tolist())

    def compute(self):
        """Compute all metrics."""
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)

        # Overall metrics
        accuracy = accuracy_score(labels, preds)

        # Per-class metrics
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        f1_per_class = f1_score(labels, preds, average=None, zero_division=0)

        # Macro averages
        precision_macro = precision_score(labels, preds, average='macro', zero_division=0)
        recall_macro = recall_score(labels, preds, average='macro', zero_division=0)
        f1_macro = f1_score(labels, preds, average='macro', zero_division=0)

        # Confusion matrix
        conf_matrix = confusion_matrix(labels, preds)

        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_per_class': {self.class_names[i]: precision_per_class[i] for i in range(self.num_classes)},
            'recall_per_class': {self.class_names[i]: recall_per_class[i] for i in range(self.num_classes)},
            'f1_per_class': {self.class_names[i]: f1_per_class[i] for i in range(self.num_classes)},
            'confusion_matrix': conf_matrix
        }

    def print_metrics(self, metrics):
        """Pretty print metrics."""
        print("\n" + "="*60)
        print("CORAL HEALTH CLASSIFICATION METRICS")
        print("="*60)
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro Precision:  {metrics['precision_macro']:.4f}")
        print(f"Macro Recall:     {metrics['recall_macro']:.4f}")
        print(f"Macro F1-Score:   {metrics['f1_macro']:.4f}")

        print("\nPer-Class Metrics:")
        print("-"*60)
        print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*60)
        for cls in self.class_names:
            print(f"{cls:<15} {metrics['precision_per_class'][cls]:<12.4f} "
                  f"{metrics['recall_per_class'][cls]:<12.4f} "
                  f"{metrics['f1_per_class'][cls]:<12.4f}")

        print("\nConfusion Matrix:")
        print("-"*60)
        print("Rows: True labels, Columns: Predicted labels")
        print(f"         {' '.join([f'{cls[:8]:>8}' for cls in self.class_names])}")
        for i, cls in enumerate(self.class_names):
            row = metrics['confusion_matrix'][i]
            print(f"{cls[:8]:>8} {' '.join([f'{val:>8}' for val in row])}")
        print("="*60 + "\n")
