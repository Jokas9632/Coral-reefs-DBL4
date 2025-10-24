#Dataset loader
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

def _read_rgb(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))

def _read_rgb_mask(p: Path) -> np.ndarray:
    return np.array(Image.open(p).convert("RGB"))

def _resize_np(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    return np.array(Image.fromarray(img).resize((size[1], size[0]), resample=Image.BILINEAR))

def _color_to_binary_and_label(mask_rgb: np.ndarray, label_mode: str = "any_bleached") -> tuple[np.ndarray, int]:
    #Converts masks to right colours for classification aid.
    r, g, b = mask_rgb[..., 0], mask_rgb[..., 1], mask_rgb[..., 2]
    red   = (r > 150) & (g < 80) & (b < 80)
    blue  = (b > 150) & (r < 80) & (g < 120)
    coral = red | blue
    if label_mode == "any_bleached":
        label = int(red.any())
    else:  
        label = int(red.sum() > blue.sum())
    return coral.astype(np.uint8), label

class CoralMaskClsDataset(Dataset):
    def __init__(self, root: str, img_size: Tuple[int,int]=(640,640),
                 label_mode: str = "any_bleached",
                 image_suffixes=(".jpg", ".jpeg", ".png")):
        self.root = Path(root)
        self.img_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.size = img_size
        self.label_mode = label_mode
        self.items = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in image_suffixes])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_p = self.items[idx]
        msk_p = self.mask_dir / (img_p.stem + ".png")
        img = _read_rgb(img_p)
        msk = _read_rgb_mask(msk_p)

        img = _resize_np(img, self.size)
        msk = _resize_np(msk, self.size)

        coral_bin, label = _color_to_binary_and_label(msk, label_mode=self.label_mode)

        img_t = TF.to_tensor(Image.fromarray(img))                         # CxHxW float [0,1]
        mask_t = torch.from_numpy(coral_bin[None].astype(np.float32))      # 1xHxW float {0,1}
        label_t = torch.tensor(label, dtype=torch.long)                    # scalar {0,1}

        return {"image": img_t, "mask": mask_t, "label": label_t, "id": img_p.stem}
