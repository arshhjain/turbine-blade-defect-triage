"""
Data Augmentation Pipeline using Albumentations for Wind Turbine Blade Defect Classification.

This module implements a heavy data augmentation pipeline to mitigate data scarcity,
small sample sizes, and environmental variations in drone RGB imagery (lighting,
weather, flight vibrations, and camera angles).
"""

from typing import Tuple, List, Optional
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


def get_train_transforms(
    image_size: int = 224,
    heavy: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Build Albumentations heavy augmentation pipeline for training drone imagery.

    Args:
        image_size: Dynamic image resolution for progressive resolution training.
        heavy: If True, applies extensive spatial and photometrical distortions.
        mean: ImageNet channel means.
        std: ImageNet channel standard deviations.

    Returns:
        Albumentations Compose pipeline.
    """
    hole_min = max(8, int(image_size * 0.04))
    hole_max = max(16, int(image_size * 0.12))

    transforms: List[A.BasicTransform] = [
        # 1. Geometric Rotations and Invariance to Drone Flight Angles
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.85, 1.15),
            translate_percent=(-0.06, 0.06),
            rotate=(-45, 45),
            shear=(-8, 8),
            p=0.7,
        ),
    ]

    if heavy:
        transforms.extend([
            # 2. Localized Blurring (simulating drone flight motion and lens vibration)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=0.6),
                A.MotionBlur(blur_limit=(3, 7), p=0.4),
                A.MedianBlur(blur_limit=5, p=0.3),
            ], p=0.4),

            # 3. Systematic Color Shifts & Lighting Inconsistencies (sunlight, shadows, overcast sky)
            A.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.25,
                hue=0.10,
                p=0.6
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.25,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=15,
                sat_shift_limit=25,
                val_shift_limit=20,
                p=0.4
            ),

            # 4. Sensor Artifact Simulation (Cutout / Dropout)
            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(hole_min, hole_max),
                hole_width_range=(hole_min, hole_max),
                fill=0,
                p=0.3
            ),
        ])

    # 6. Progressive Resolution Resizing & Tensor Normalization
    transforms.extend([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

    return A.Compose(transforms)


def get_val_transforms(
    image_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    Deterministic validation/inference transform pipeline.

    Args:
        image_size: Input resolution to evaluate.
        mean: ImageNet channel means.
        std: ImageNet channel standard deviations.

    Returns:
        Albumentations Compose pipeline.
    """
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def denormalize_image(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """
    Convert normalized PyTorch image tensor (C, H, W) back to displayable RGB uint8 array (H, W, C).
    Useful for Grad-CAM overlay and inspection plots.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(std) + np.array(mean)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img
