#Train Coral with YoloV11 Classifier pyfile
import argparse, os, random, numpy as np, torch
from torch.utils.data import DataLoader
import torch.optim as optim

from data.dataset import CoralMaskClsDataset
from models.coral_classifier import CoralClassifier
from engine.trainer import Trainer

def seed_all(seed: int = 0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, default="data/train", help="root with images/ and masks/")
    ap.add_argument("--val_root", type=str, default="data/val", help="root with images/ and masks/")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--device", type=str, default="auto", help="'auto'|'cpu'|'cuda'")
    ap.add_argument("--weights", type=str, default="yolov11s.pt", help="Ultralytics YOLOv11 weights")
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--label_mode", type=str, default="any_bleached", choices=["any_bleached", "majority"])
    return ap.parse_args()

def main():
    args = parse_args()
    seed_all(0)

    train_ds = CoralMaskClsDataset(args.train_root, img_size=(args.imgsz, args.imgsz), label_mode=args.label_mode)
    val_ds   = CoralMaskClsDataset(args.val_root,   img_size=(args.imgsz, args.imgsz), label_mode=args.label_mode)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=False)

    model = CoralClassifier(
        yolo_weights=args.weights,
        num_classes=2,
        mask_channels=64,
        freeze_backbone=args.freeze_backbone
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    trainer = Trainer(model, optimizer, device=args.device)  # auto-selects cuda if available
    trainer.fit(train_loader, val_loader, epochs=args.epochs, verbose=True)

if __name__ == "__main__":
    main()
