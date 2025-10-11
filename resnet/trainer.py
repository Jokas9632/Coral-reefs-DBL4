from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from .metrics import CoralMetrics


class CoralTrainer:
    """
    Minimal binary trainer with:
      - BCEWithLogitsLoss (+ optional pos_weight = N_neg / N_pos)
      - AdamW defaults (3e-4, wd=0.05)
      - Progress bar
      - Per-epoch eval + printed metrics
    """
    def __init__(self, model: nn.Module, device: str = 'cuda', threshold: float = 0.5):
        self.model = model
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.threshold = float(threshold)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        optimizer: Optional[torch.optim.Optimizer] = None,
        pos_weight: Optional[float] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, list]:

        # Binary loss (always): BCEWithLogitsLoss with optional positive reweighting
        if pos_weight is not None:
            weight_tensor = torch.tensor([float(pos_weight)], device=self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)
        else:
            criterion = nn.BCEWithLogitsLoss()

        # (5) Optimizer: AdamW robust defaults
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=3e-4, weight_decay=0.05, betas=(0.9, 0.999)
            )

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(1, epochs + 1):
            self.model.train()
            running = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]")
            for images, targets in pbar:
                images = images.to(self.device, non_blocking=True)
                # targets expected as 0/1 integers; cast to float column vector for BCE
                targets = targets.to(self.device, non_blocking=True).float().unsqueeze(1)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(images)
                loss = criterion(logits, targets)
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
                scheduler.step()

        return history

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, criterion=None):
        self.model.eval()
        running = 0.0
        metrics = CoralMetrics()

        for images, targets in tqdm(loader, desc="Eval"):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images)
            if criterion is not None:
                running += float(criterion(logits, targets.float().unsqueeze(1)).item())

            probs = torch.sigmoid(logits).squeeze(1)
            preds = (probs >= self.threshold).long()
            metrics.update(preds.cpu(), targets.cpu())

        return running / max(1, len(loader)), metrics.compute()
