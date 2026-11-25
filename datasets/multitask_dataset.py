"""
PyTorch Dataset Loader for Wind Turbine Blade Multi-Task Defect & Severity Prediction.

Loads from leak-free split manifest (data/splits/leak_free_split_manifest.json)
and applies data augmentation strictly to the training split.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASSES = [
    "crack",
    "craze_crack",
    "delamination",
    "erosion",
    "lightning_damage",
    "surface_damage"
]

CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}


class TurbineBladeMultiTaskDataset(Dataset):
    """
    Multi-Task Wind Turbine Blade Dataset for Defect Type & Severity Prediction.
    """
    def __init__(
        self,
        split: str = "train",
        manifest_path: Optional[str] = None,
        transform: Optional[Callable] = None,
        image_size: int = 384
    ):
        super().__init__()
        self.split = split
        self.image_size = image_size
        self.transform = transform

        if manifest_path is None:
            manifest_path = PROJECT_ROOT / "data" / "splits" / "leak_free_split_manifest.json"
        else:
            manifest_path = Path(manifest_path)

        if not manifest_path.exists():
            raise FileNotFoundError(f"Split manifest not found at: {manifest_path}. Run scripts/make_group_splits.py first.")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        if split not in manifest:
            raise ValueError(f"Split '{split}' not found in manifest. Available: {list(manifest.keys())}")

        self.samples = manifest[split]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        record = self.samples[idx]
        img_full_path = PROJECT_ROOT / record["image_path"]

        with Image.open(str(img_full_path)) as pil_img:
            pil_rgb = pil_img.convert("RGB")
            resized = pil_rgb.resize((self.image_size, self.image_size), Image.BILINEAR)
            image_rgb = np.array(resized, dtype=np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=image_rgb)
            image_tensor = augmented["image"]
        else:
            # Fallback simple resize and normalization
            resized = cv2.resize(image_rgb, (self.image_size, self.image_size)).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(resized).permute(2, 0, 1)

        target_type = torch.tensor(CLASS_TO_IDX[record["dominant_class"]], dtype=torch.long)
        target_severity = torch.tensor(float(record["severity_level"]), dtype=torch.float32)

        meta = {
            "image_id": record["image_id"],
            "image_path": record["image_path"],
            "group_id": record["group_id"],
            "dominant_class": record["dominant_class"],
            "severity_level": record["severity_level"]
        }

        return image_tensor, target_type, target_severity, meta

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute inverse square-root class frequency weights:
        alpha_c = 1.0 / sqrt(N_c), normalized such that sum(alpha_c) = num_classes.
        """
        counts = np.zeros(len(CLASSES), dtype=np.float32)
        for s in self.samples:
            cls_idx = CLASS_TO_IDX[s["dominant_class"]]
            counts[cls_idx] += 1.0

        # Prevent division by zero
        counts = np.maximum(counts, 1.0)
        inv_sqrt = 1.0 / np.sqrt(counts)
        normalized_weights = inv_sqrt / np.sum(inv_sqrt) * len(CLASSES)
        return torch.tensor(normalized_weights, dtype=torch.float32)
