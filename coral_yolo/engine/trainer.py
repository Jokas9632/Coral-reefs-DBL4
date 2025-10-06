"""Trainer class for coral bleaching classification (with per-class metrics)."""

from typing import Dict, Any
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from coral_yolo.losses.classification_loss import CoralClassificationLoss
from coral_yolo.engine.metrics import ClsPRF1


# --- automatic collate with max_size cap -----------------------------------
def _auto_pad_collate(batch, max_size=384):
    """Pads variable-sized images/masks; supports dict or tuple outputs, capped to max_size."""
    if isinstance(batch[0], dict):
        imgs = [b["image"] for b in batch]
        masks = [b["mask"] for b in batch]
        labels = [b.get("label", torch.tensor(0)) for b in batch]
    else:
        imgs, masks = zip(*batch)
        labels = [torch.tensor(0) for _ in imgs]

    imgs = [F.interpolate(i.unsqueeze(0), size=(max_size, max_size),
                          mode="bilinear", align_corners=False).squeeze(0)
            for i in imgs]
    masks = [F.interpolate(m.unsqueeze(0), size=(max_size, max_size),
                           mode="nearest").squeeze(0)
             for m in masks]

    imgs = torch.stack(imgs)
    masks = torch.stack(masks)
    labels = torch.tensor([int(l) for l in labels], dtype=torch.long)

    return {"image": imgs, "mask": masks, "label": labels}


# --- TRAINER ---------------------------------------------------------------
class Trainer:
    def __init__(self, model, optimizer, device="auto", criterion=None, metric=None, grad_accum_steps=1):
        self.model = model
        self.optimizer = optimizer

        # Device setup
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
        self.scaler = GradScaler(enabled=(self.device.type == "cuda"), device=self.device.type)

        # Class labels for printing
        self.class_names = ["Healthy", "Bleached"]

    # -----------------------------------------------------------------------
    def _run_epoch(self, loader, train=True, epoch=0, total_epochs=0):
        """Run one epoch and return metrics dict."""
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

            with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"):
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

            # Free memory
            del images, masks, labels, logits, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        stats = self.metric.compute()
        stats["loss"] = total_loss / max(n, 1)
        return stats

    # -----------------------------------------------------------------------
        # -----------------------------------------------------------------------
    def fit(self, train_loader, val_loader=None, epochs=10, verbose=True, save_dir="checkpoints"):
        """Train the model, print per-class metrics, and save checkpoints."""

        import os
        from pathlib import Path

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        best_f1 = 0.0  # track best validation F1

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

            # --- Print metrics ---
            if verbose:
                msg = (
                    f"✅ Epoch {ep:03d}/{epochs} | "
                    f"Train loss {tr['loss']:.4f} | Acc={tr.get('accuracy', 0):.3f} | F1={tr.get('f1', 0):.3f}"
                )
                if vl:
                    msg += (
                        f" || Val loss {vl['loss']:.4f} | Acc={vl.get('accuracy', 0):.3f} | F1={vl.get('f1', 0):.3f}"
                    )
                print(msg)

                # Per-class printout
                if "per_class" in vl:
                    pcs = vl["per_class"]
                    print("📈 Per-class validation metrics:")
                    for i, cname in enumerate(self.class_names):
                        print(f"   {cname:10s} | Prec={pcs['precision'][i]:.3f} | "
                              f"Rec={pcs['recall'][i]:.3f} | F1={pcs['f1'][i]:.3f}")
                print("-" * 80)

            # --- Save checkpoints ---
            ckpt_path = os.path.join(save_dir, f"epoch_{ep:03d}.pt")
            torch.save({
                "epoch": ep,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_metrics": tr,
                "val_metrics": vl
            }, ckpt_path)

            print(f"💾 Model saved to {ckpt_path}")

            # --- Track and save best model (based on val F1) ---
            val_f1 = vl.get("f1", 0.0)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_path = os.path.join(save_dir, "best_model.pt")
                torch.save(self.model.state_dict(), best_path)
                print(f"🏆 New best model saved! (Val F1={best_f1:.3f})")

