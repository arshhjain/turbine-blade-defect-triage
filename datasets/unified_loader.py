"""
Unified Dataset Loader for Wind Turbine Blade Defect Classification.

Integrates:
1. Figshare "Multiclass Dataset for Intelligent Detection of Wind Turbine Blade Defects Using Drone Imagery"
2. Roboflow Object Detection exports (YOLO txt format or COCO format with crop extraction or image-level label assignment)
3. Standard directory-per-class structures and CSV manifests
"""

import os
import glob
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Callable, Any
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# Canonical class normalization mapping
CANONICAL_CLASS_MAP = {
    # Cracking variants
    "crack": "cracking",
    "cracks": "cracking",
    "cracking": "cracking",
    "craze": "cracking",
    "fracture": "cracking",
    "structural_crack": "cracking",
    "hairline_crack": "cracking",

    # Erosion variants
    "erosion": "erosion",
    "leading_edge_erosion": "erosion",
    "leading-edge-erosion": "erosion",
    "pitting": "erosion",
    "surface_wear": "erosion",
    "coating_peel": "erosion",
    "corrosion": "erosion",
    "surface_injure": "erosion",

    # Delamination variants
    "delamination": "delamination",
    "delam": "delamination",
    "peeling": "delamination",
    "skin_disbonding": "delamination",
    "blister": "delamination",
    "hide_craze": "delamination",

    # Normal / Pristine blade variants
    "normal": "normal",
    "clean": "normal",
    "healthy": "normal",
    "blade": "normal",
    "intact": "normal",
    "background": "normal",
}


class BladeSample:
    """Represents a single wind turbine blade inspection sample."""
    def __init__(
        self,
        image_path: str,
        label: str,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        source: str = "generic",
    ):
        self.image_path = image_path
        self.label = label
        self.bbox = bbox  # (x1, y1, x2, y2) in normalized or pixel coordinates
        self.source = source


class BladeDefectDataset(Dataset):
    """
    Unified PyTorch Dataset for Wind Turbine Blade Defect Classification.
    """
    def __init__(
        self,
        samples: List[BladeSample],
        class_to_idx: Dict[str, int],
        transform: Optional[Callable] = None,
        crop_to_bbox: bool = False,
        crop_margin: float = 0.15,
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.transform = transform
        self.crop_to_bbox = crop_to_bbox
        self.crop_margin = crop_margin

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        sample = self.samples[idx]
        image_path = sample.image_path

        # Load RGB image via OpenCV
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image at: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        # Optional localized defect crop if bounding box is provided (Roboflow detection integration)
        if self.crop_to_bbox and sample.bbox is not None:
            bx1, by1, bx2, by2 = sample.bbox
            if 0.0 <= bx1 <= 1.0 and 0.0 <= bx2 <= 1.0:
                # Convert normalized coords to pixel coords
                bx1, by1, bx2, by2 = int(bx1 * w), int(by1 * h), int(bx2 * w), int(by2 * h)
            else:
                bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2)

            # Apply aerodynamic context margin
            pad_w = int((bx2 - bx1) * self.crop_margin)
            pad_h = int((by2 - by1) * self.crop_margin)
            x1 = max(0, bx1 - pad_w)
            y1 = max(0, by1 - pad_h)
            x2 = min(w, bx2 + pad_w)
            y2 = min(h, by2 + pad_h)

            if (x2 - x1) > 10 and (y2 - y1) > 10:
                image = image[y1:y2, x1:x2]

        # Apply Albumentations augmentations
        if self.transform is not None:
            augmented = self.transform(image=image)
            image_tensor = augmented["image"]
        else:
            # Fallback tensor conversion
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        label_idx = self.class_to_idx[sample.label]
        return image_tensor, label_idx, sample.image_path


def parse_figshare_dataset(figshare_root: str, target_classes: List[str]) -> List[BladeSample]:
    """
    Parse the Figshare 'Multiclass Dataset for Intelligent Detection of Wind Turbine Blade Defects Using Drone Imagery'.
    Supports:
    1. Pascal VOC XML Annotations/ and JPEGImages/ (Official Springer Nature release format)
    2. Direct category folders (cracking/, erosion/, delamination/)
    3. Flat folder with annotation CSV metadata
    """
    import xml.etree.ElementTree as ET
    samples: List[BladeSample] = []
    root = Path(figshare_root)
    if not root.exists():
        return samples

    # Search recursively for Pascal VOC Annotations and JPEGImages directories
    ann_dirs = list(root.glob("**/Annotations"))
    for ann_dir in ann_dirs:
        img_dir = ann_dir.parent / "JPEGImages"
        if not img_dir.exists():
            img_dir = ann_dir.parent / "images"
        if img_dir.exists():
            for xml_file in ann_dir.glob("*.xml"):
                try:
                    tree = ET.parse(xml_file)
                    xml_root = tree.getroot()
                    filename_elem = xml_root.find("filename")
                    img_filename = filename_elem.text if filename_elem is not None else f"{xml_file.stem}.jpg"
                    img_path = img_dir / img_filename
                    if not img_path.exists():
                        # Try case-insensitive or extension alternatives
                        candidates = list(img_dir.glob(f"{xml_file.stem}.*"))
                        if candidates:
                            img_path = candidates[0]

                    if not img_path.exists():
                        continue

                    # Parse objects / defects
                    objects = xml_root.findall("object")
                    for obj in objects:
                        name_elem = obj.find("name")
                        if name_elem is not None and name_elem.text:
                            raw_name = name_elem.text.strip().lower()
                            norm_name = CANONICAL_CLASS_MAP.get(raw_name, raw_name)
                            if norm_name in target_classes:
                                bnd = obj.find("bndbox")
                                bbox = None
                                if bnd is not None:
                                    try:
                                        xmin = float(bnd.find("xmin").text)
                                        ymin = float(bnd.find("ymin").text)
                                        xmax = float(bnd.find("xmax").text)
                                        ymax = float(bnd.find("ymax").text)
                                        bbox = (xmin, ymin, xmax, ymax)
                                    except Exception:
                                        bbox = None

                                samples.append(BladeSample(
                                    image_path=str(img_path),
                                    label=norm_name,
                                    bbox=bbox,
                                    source="figshare_voc"
                                ))
                except Exception:
                    continue

    # Also check for direct category folders
    for entry in root.iterdir():
        if entry.is_dir():
            folder_name = entry.name.lower()
            norm_label = CANONICAL_CLASS_MAP.get(folder_name, folder_name)
            if norm_label in target_classes:
                for img_ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
                    for img_file in entry.glob(img_ext):
                        samples.append(BladeSample(
                            image_path=str(img_file),
                            label=norm_label,
                            source="figshare_dir"
                        ))

    # Also check if images are flat with an annotation csv
    csv_candidates = list(root.glob("*.csv")) + list(root.glob("*.txt"))
    for csv_file in csv_candidates:
        try:
            df = pd.read_csv(csv_file)
            col_candidates_img = [c for c in df.columns if any(k in c.lower() for k in ["image", "filename", "file", "img"])]
            col_candidates_lbl = [c for c in df.columns if any(k in c.lower() for k in ["label", "class", "defect", "category"])]
            if col_candidates_img and col_candidates_lbl:
                img_col = col_candidates_img[0]
                lbl_col = col_candidates_lbl[0]
                for _, row in df.iterrows():
                    raw_lbl = str(row[lbl_col]).strip().lower()
                    norm_label = CANONICAL_CLASS_MAP.get(raw_lbl, raw_lbl)
                    if norm_label in target_classes:
                        img_path = root / str(row[img_col])
                        if not img_path.exists():
                            img_path = root / "images" / str(row[img_col])
                        if img_path.exists():
                            samples.append(BladeSample(
                                image_path=str(img_path),
                                label=norm_label,
                                source="figshare_csv"
                            ))
        except Exception:
            pass

    return samples


def parse_roboflow_yolo_dataset(
    roboflow_root: str,
    target_classes: List[str],
    split: str = "train"
) -> List[BladeSample]:
    """
    Parse Roboflow YOLO-format Wind Turbine Blade Defect datasets.
    Reads images/ and labels/ directories with class bounding boxes.
    """
    samples: List[BladeSample] = []
    root = Path(roboflow_root)
    if not root.exists():
        return samples

    # Look for split directory (e.g., train, valid, test) or directly in root
    split_dir = root / split if (root / split).exists() else root
    img_dir = split_dir / "images" if (split_dir / "images").exists() else split_dir
    lbl_dir = split_dir / "labels" if (split_dir / "labels").exists() else split_dir

    # Check for data.yaml or notes.json to read class names
    roboflow_class_names: List[str] = []
    yaml_candidates = list(root.glob("*.yaml")) + list(root.glob("*.yml"))
    if yaml_candidates:
        import yaml
        try:
            with open(yaml_candidates[0], "r", encoding="utf-8") as f:
                data_yaml = yaml.safe_load(f)
                if "names" in data_yaml:
                    if isinstance(data_yaml["names"], list):
                        roboflow_class_names = [str(x).lower() for x in data_yaml["names"]]
                    elif isinstance(data_yaml["names"], dict):
                        roboflow_class_names = [str(data_yaml["names"][k]).lower() for k in sorted(data_yaml["names"].keys())]
        except Exception:
            pass

    if not roboflow_class_names:
        # Default Roboflow defect classes for wind turbine blade benchmarks
        roboflow_class_names = ["cracking", "erosion", "delamination", "normal"]

    # Search images
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
        image_paths.extend(img_dir.glob(ext))

    for img_path in image_paths:
        # Check label file
        txt_path = lbl_dir / f"{img_path.stem}.txt"
        if txt_path.exists():
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            if not lines:
                # No defects detected -> normal blade
                if "normal" in target_classes:
                    samples.append(BladeSample(
                        image_path=str(img_path),
                        label="normal",
                        source="roboflow_clean"
                    ))
            else:
                # Parse each detection box
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        cls_idx = int(parts[0])
                        if cls_idx < len(roboflow_class_names):
                            raw_cls = roboflow_class_names[cls_idx]
                            norm_cls = CANONICAL_CLASS_MAP.get(raw_cls, raw_cls)
                            if norm_cls in target_classes:
                                xc, yc, bw, bh = map(float, parts[1:5])
                                x1 = max(0.0, xc - bw / 2.0)
                                y1 = max(0.0, yc - bh / 2.0)
                                x2 = min(1.0, xc + bw / 2.0)
                                y2 = min(1.0, yc + bh / 2.0)
                                samples.append(BladeSample(
                                    image_path=str(img_path),
                                    label=norm_cls,
                                    bbox=(x1, y1, x2, y2),
                                    source="roboflow_box"
                                ))
        else:
            # If no label file exists, check parent folder name
            folder_cls = img_path.parent.name.lower()
            norm_cls = CANONICAL_CLASS_MAP.get(folder_cls, folder_cls)
            if norm_cls in target_classes:
                samples.append(BladeSample(
                    image_path=str(img_path),
                    label=norm_cls,
                    source="roboflow_dir"
                ))

    return samples


def parse_generic_directory(data_dir: str, target_classes: List[str]) -> List[BladeSample]:
    """
    Parse standard directory hierarchy: data_dir/<class_name>/<image>.
    """
    samples: List[BladeSample] = []
    root = Path(data_dir)
    if not root.exists():
        return samples

    for class_folder in root.iterdir():
        if class_folder.is_dir():
            norm_cls = CANONICAL_CLASS_MAP.get(class_folder.name.lower(), class_folder.name.lower())
            if norm_cls in target_classes:
                for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
                    for img_file in class_folder.glob(ext):
                        samples.append(BladeSample(
                            image_path=str(img_file),
                            label=norm_cls,
                            source="directory"
                        ))

    return samples


def build_dataloaders(
    config: Dict[str, Any],
    resolution: int = 224,
    transform_train: Optional[Callable] = None,
    transform_val: Optional[Callable] = None,
    crop_to_bbox: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """
    Build unified train, validation, and test dataloaders from all configured sources.
    """
    target_classes = config["dataset"]["classes"]
    class_to_idx = {cls: idx for idx, cls in enumerate(target_classes)}

    all_samples: List[BladeSample] = []

    # 1. Figshare integration
    figshare_dir = config["dataset"].get("figshare_dir", "data/figshare")
    figshare_samples = parse_figshare_dataset(figshare_dir, target_classes)
    all_samples.extend(figshare_samples)

    # 2. Roboflow integration
    roboflow_dir = config["dataset"].get("roboflow_dir", "data/roboflow")
    roboflow_samples = parse_roboflow_yolo_dataset(roboflow_dir, target_classes)
    all_samples.extend(roboflow_samples)

    # 3. Synthetic / Benchmark generator directory
    synthetic_dir = config["dataset"].get("synthetic_dir", "data/synthetic")
    synthetic_samples = parse_generic_directory(synthetic_dir, target_classes)
    all_samples.extend(synthetic_samples)

    # 4. Standard data dir
    data_dir = config["dataset"].get("data_dir", "data")
    generic_samples = parse_generic_directory(data_dir, target_classes)
    all_samples.extend(generic_samples)

    # Deduplicate by image_path and bbox
    unique_samples: List[BladeSample] = []
    seen = set()
    for s in all_samples:
        key = (s.image_path, s.label, str(s.bbox))
        if key not in seen:
            seen.add(key)
            unique_samples.append(s)

    if len(unique_samples) == 0:
        raise ValueError(
            f"No valid blade defect images found in configured directories. "
            f"Run mock generator or point config to Figshare/Roboflow paths."
        )

    # Stratified split into train, val, test
    labels = [s.label for s in unique_samples]
    val_ratio = config["dataset"].get("val_split", 0.2)
    test_ratio = config["dataset"].get("test_split", 0.1)

    # Ensure each class has enough samples for splitting
    train_samples, temp_samples = train_test_split(
        unique_samples,
        test_size=(val_ratio + test_ratio),
        random_state=config["project"]["random_seed"],
        stratify=labels if len(set(labels)) > 1 else None
    )

    temp_labels = [s.label for s in temp_samples]
    val_size_adjusted = val_ratio / (val_ratio + test_ratio)
    val_samples, test_samples = train_test_split(
        temp_samples,
        test_size=(1.0 - val_size_adjusted),
        random_state=config["project"]["random_seed"],
        stratify=temp_labels if len(set(temp_labels)) > 1 else None
    )

    # Build Dataset objects
    train_dataset = BladeDefectDataset(
        train_samples, class_to_idx, transform=transform_train, crop_to_bbox=crop_to_bbox
    )
    val_dataset = BladeDefectDataset(
        val_samples, class_to_idx, transform=transform_val, crop_to_bbox=crop_to_bbox
    )
    test_dataset = BladeDefectDataset(
        test_samples, class_to_idx, transform=transform_val, crop_to_bbox=crop_to_bbox
    )

    batch_size = config["progressive_training"]["stages"][0]["batch_size"]
    num_workers = config["dataset"].get("num_workers", 0)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_to_idx
