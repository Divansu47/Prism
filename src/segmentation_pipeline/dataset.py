import os
from pathlib import Path
from typing import Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class FoodSegmentationDataset(Dataset):
    def __init__(self, image_dir: str, mask_dir: str, labels_csv: str, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.df = pd.read_csv(labels_csv)
        self.class_names = [c for c in self.df.columns if c not in {"image_id", "image_path", "mask_path"}]

    @staticmethod
    def _resolve_path(base_dir: Path, raw_path: str) -> Path:
        raw = Path(str(raw_path))
        candidates = [raw]
        candidates.append(base_dir / raw)
        candidates.append(base_dir / raw.name)
        if raw.parts and raw.parts[0] == base_dir.name:
            candidates.append(base_dir / Path(*raw.parts[1:]))
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not resolve path '{raw_path}' under {base_dir}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image_path = self._resolve_path(self.image_dir, row["image_path"])
        mask_path = self._resolve_path(self.mask_dir, row["mask_path"])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask.squeeze(0)
        if mask.dim() == 2:
            h, w = mask.shape
            mask = mask.unsqueeze(0)
        else:
            h, w = mask.shape[-2], mask.shape[-1]

        target = torch.zeros((len(self.class_names), h, w), dtype=torch.float32)
        for i, _ in enumerate(self.class_names):
            class_mask = (mask[0] == (i + 1)).float()
            target[i] = class_mask

        return image, target
