"""Trainer class for coral bleaching classification."""

from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
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
    def __init__(self, model, optimizer, device="auto", criterion=None, metric=None, grad_accum_steps=1):
        self.model = model
        self.optimizer = optimizer

        # ---- device selection (honors "cuda", "cuda:0", "cpu", or auto) ----
        if isinstance(device, torch.device):
            self.device = device
        elif device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ---- move model once; prefer channels_last for conv perf ----
        self.model = self.model.to(self.device).to(memory_format=torch.channels_last)

        # ---- loss & metric (this fixes your AttributeError) ----
        self.criterion = criterion if criterion is not None else CoralClassificationLoss()
        self.metric    = metric    if metric    is not None else ClsPRF1()

        # ---- AMP scaler (enabled only on CUDA) ----
        self._scaler = GradScaler(enabled=(self.device.type == "cuda"))

        # ---- optional: gradient accumulation ----
        self.grad_accum_steps = int(grad_accum_steps)




    def _run_epoch(self, loader, train: bool):
        self.model.train(mode=train)
        self.metric.reset()
        total_loss, n = 0.0, 0

        for i, batch in enumerate(loader):
            images = batch["image"].to(self.device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            masks  = batch["mask"].to(self.device,  non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            use_amp = (self.device.type == "cuda")

            # zero every grad_accum_steps
            if train and (i % self.grad_accum_steps == 0):
                self.optimizer.zero_grad(set_to_none=True)

            if train:
                if use_amp:
                    with autocast(dtype=torch.float16):
                        logits = self.model(images, masks)
                        loss = self.criterion(logits, labels) / self.grad_accum_steps
                    self._scaler.scale(loss).backward()
                    if (i + 1) % self.grad_accum_steps == 0:
                        self._scaler.step(self.optimizer)
                        self._scaler.update()
                else:
                    logits = self.model(images, masks)
                    loss = self.criterion(logits, labels) / self.grad_accum_steps
                    loss.backward()
                    if (i + 1) % self.grad_accum_steps == 0:
                        self.optimizer.step()
            else:
                with torch.no_grad():
                    if use_amp:
                        with autocast(dtype=torch.float16):
                            logits = self.model(images, masks)
                            loss = self.criterion(logits, labels)
                    else:
                        logits = self.model(images, masks)
                        loss = self.criterion(logits, labels)

        stats = self.metric.compute()
        stats["loss"] = total_loss / max(n, 1)
        return stats

    def fit(self, train_loader, val_loader=None, epochs=10, verbose=True):
        """Trains for N epochs with validation after each epoch."""

        # --- FIX: enforce pad collate explicitly ---
        def _wrap_loader(loader, shuffle=False):
            return DataLoader(
                loader.dataset,
                batch_size=loader.batch_size,
                shuffle=shuffle,
                num_workers=loader.num_workers,
                collate_fn=_auto_pad_collate,          # keep your existing collate
                pin_memory=(self.device.type == "cuda"),
                persistent_workers=(loader.num_workers > 0),
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
