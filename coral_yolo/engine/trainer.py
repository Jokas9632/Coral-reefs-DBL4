"""Trainer class for coral bleaching classification."""

from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from coral_yolo.losses.classification_loss import CoralClassificationLoss
from coral_yolo.engine.metrics import ClsPRF1


# --- NEW: automatic collate that handles dict or tuple data -----------------
def _auto_pad_collate(batch):
    """Pads variable-sized images/masks; supports dict or tuple outputs."""
    # Detect sample type
    if isinstance(batch[0], dict):
        imgs  = [b["image"] for b in batch]
        masks = [b["mask"]  for b in batch]
        labels = [b.get("label", torch.tensor(0)) for b in batch]
    else:  # tuple form (img, mask)
        imgs, masks = zip(*batch)
        labels = [torch.tensor(0) for _ in imgs]

    max_h = max(i.shape[1] for i in imgs)
    max_w = max(i.shape[2] for i in imgs)

    def _pad(t):
        pad_h = max_h - t.shape[1]
        pad_w = max_w - t.shape[2]
        return F.pad(t, (0, pad_w, 0, pad_h))

    imgs  = torch.stack([_pad(i) for i in imgs])
    masks = torch.stack([_pad(m) for m in masks])
    labels = torch.tensor([int(l) for l in labels], dtype=torch.long)

    return {"image": imgs, "mask": masks, "label": labels}


# --- TRAINER ----------------------------------------------------------------
class Trainer:
    """Simple CPU/GPU trainer with automatic padding and metric logging."""

    def __init__(self, model, optimizer, device="auto"):
        self.model = model
        self.optimizer = optimizer
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        )
        self.criterion = CoralClassificationLoss()
        self.metric = ClsPRF1()
        self.model.to(self.device)

    def _run_epoch(self, loader: DataLoader, train: bool = True) -> Dict[str, Any]:
        """Runs one epoch and returns averaged loss + metrics."""
        self.model.train(mode=train)
        self.metric.reset()
        total_loss, n = 0.0, 0

        for batch in loader:
            images = batch["image"].to(self.device)
            masks  = batch["mask"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(images, masks)
            loss = self.criterion(logits, labels)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

            self.metric.update(logits.detach(), labels.detach())
            total_loss += loss.item()
            n += 1

        stats = self.metric.compute()
        stats["loss"] = total_loss / max(n, 1)
        return stats

    def fit(self, train_loader, val_loader=None, epochs=10, verbose=True):
        """Trains for N epochs with validation after each epoch."""

        # --- FIX: enforce pad collate explicitly ---
        def _wrap_loader(loader, shuffle=False):
            if not hasattr(loader, "collate_fn") or loader.collate_fn is None:
                # if the loader has no collate, wrap it
                return DataLoader(
                    loader.dataset,
                    batch_size=loader.batch_size,
                    shuffle=shuffle,
                    num_workers=loader.num_workers,
                    collate_fn=_auto_pad_collate,
                )
            # if collate_fn exists but not ours, still override (to be safe)
            return DataLoader(
                loader.dataset,
                batch_size=loader.batch_size,
                shuffle=shuffle,
                num_workers=loader.num_workers,
                collate_fn=_auto_pad_collate,
            )

        train_loader = _wrap_loader(train_loader, shuffle=True)
        val_loader = _wrap_loader(val_loader, shuffle=False) if val_loader else None
        # ------------------------------------------------------------

        for ep in range(1, epochs + 1):
            tr = self._run_epoch(train_loader, train=True)
            vl = self._run_epoch(val_loader, train=False) if val_loader else {}
            if verbose:
                msg = f"Epoch {ep:03d} | Train loss {tr['loss']:.4f}"
                if vl:
                    msg += f" | Val loss {vl['loss']:.4f}"
                print(msg)
