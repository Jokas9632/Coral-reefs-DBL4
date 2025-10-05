"""Trainer class for coral bleaching classification (optimized + progress display)."""

from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm  # ✅ progress bar
from coral_yolo.losses.classification_loss import CoralClassificationLoss
from coral_yolo.engine.metrics import ClsPRF1


# --- automatic collate with size cap ---------------------------------------
def _auto_pad_collate(batch, max_size=384):
    """Pads variable-sized images/masks; supports dict or tuple outputs, capped to max_size."""
    if isinstance(batch[0], dict):
        imgs = [b["image"] for b in batch]
        masks = [b["mask"] for b in batch]
        labels = [b.get("label", torch.tensor(0)) for b in batch]
    else:
        imgs, masks = zip(*batch)
        labels = [torch.tensor(0) for _ in imgs]

    # Cap extreme sizes to avoid OOM
    imgs = [F.interpolate(i.unsqueeze(0), size=(max_size, max_size),
                          mode="bilinear", align_corners=False).squeeze(0)
            if max(i.shape[1], i.shape[2]) > max_size else i for i in imgs]
    masks = [F.interpolate(m.unsqueeze(0), size=(max_size, max_size),
                           mode="nearest").squeeze(0)
             if max(m.shape[1], m.shape[2]) > max_size else m for m in masks]

    max_h = max(i.shape[1] for i in imgs)
    max_w = max(i.shape[2] for i in imgs)

    def _pad(t):
        pad_h = max_h - t.shape[1]
        pad_w = max_w - t.shape[2]
        return F.pad(t, (0, pad_w, 0, pad_h))

    imgs = torch.stack([_pad(i) for i in imgs])
    masks = torch.stack([_pad(m) for m in masks])
    labels = torch.tensor([int(l) for l in labels], dtype=torch.long)

    return {"image": imgs, "mask": masks, "label": labels}


# --- TRAINER ----------------------------------------------------------------
class Trainer:
    def __init__(self, model, optimizer, device="auto", criterion=None, metric=None, grad_accum_steps=1):
        self.model = model
        self.optimizer = optimizer

        # Device selection
        if isinstance(device, torch.device):
            self.device = device
        elif device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = self.model.to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)

        self.criterion = criterion if criterion is not None else CoralClassificationLoss()
        self.metric = metric if metric is not None else ClsPRF1()
        self.grad_accum_steps = int(grad_accum_steps)
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"))

    # --------------------------------------------------------------------------
    def _run_epoch(self, loader, train=True, epoch=0, total_epochs=0):
        """Run one epoch with tqdm progress bar."""
        self.model.train(mode=train)
        self.metric.reset()
        total_loss, n = 0.0, 0

        desc = f"{'Train' if train else 'Val'} Epoch {epoch}/{total_epochs}"
        pbar = tqdm(loader, desc=desc, leave=False, ncols=100)

        for i, batch in enumerate(pbar):
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            if train and (i % self.grad_accum_steps == 0):
                self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(self.device.type == "cuda"), dtype=torch.float16):
                logits = self.model(images, masks)
                loss = self.criterion(logits, labels) / self.grad_accum_steps

            if train:
                self.scaler.scale(loss).backward()

                if (i + 1) % self.grad_accum_steps == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            self.metric.update(logits.detach(), labels.detach())
            total_loss += loss.item()
            n += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Free GPU memory
            del images, masks, labels, logits, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        stats = self.metric.compute()
        stats["loss"] = total_loss / max(n, 1)
        return stats

    # --------------------------------------------------------------------------
    def fit(self, train_loader, val_loader=None, epochs=10, verbose=True):
        """Trains for N epochs with validation after each epoch."""

        def _wrap(loader, shuffle=False):
            return DataLoader(
                loader.dataset,
                batch_size=loader.batch_size,
                shuffle=shuffle,
                num_workers=loader.num_workers,
                pin_memory=(self.device.type == "cuda"),
                collate_fn=_auto_pad_collate,
                persistent_workers=False,
            )

        train_loader = _wrap(train_loader, shuffle=True)
        val_loader = _wrap(val_loader, shuffle=False) if val_loader else None

        for ep in range(1, epochs + 1):
            tr = self._run_epoch(train_loader, train=True, epoch=ep, total_epochs=epochs)
            vl = self._run_epoch(val_loader, train=False, epoch=ep, total_epochs=epochs) if val_loader else {}

            if verbose:
                msg = f"✅ Epoch {ep:03d}/{epochs} | Train loss {tr['loss']:.4f}"
                if "accuracy" in tr:
                    msg += f" | Train Acc={tr['accuracy']:.3f} F1={tr['f1']:.3f}"
                if vl:
                    msg += f" | Val loss {vl['loss']:.4f}"
                    if "accuracy" in vl:
                        msg += f" | Val Acc={vl['accuracy']:.3f} F1={vl['f1']:.3f}"
                print(msg)
