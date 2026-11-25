"""
Semi-Supervised Consistency Regularization Engine for Wind Turbine Drone Imagery.

Referencing yuuStella's semi-supervised PyTorch implementation:
Leverages vast pools of unannotated drone inspection frames collected during routine
turbine blade flybys to expand effective training data beyond limited manual annotations.
"""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemiSupervisedConsistencyEngine:
    """
    Implements FixMatch-style consistency regularization with dual weak/strong
    augmentations for semi-supervised blade defect learning.
    """
    def __init__(
        self,
        model: nn.Module,
        confidence_threshold: float = 0.90,
        unlabeled_weight: float = 1.0,
    ):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.unlabeled_weight = unlabeled_weight

    def compute_unlabeled_loss(
        self,
        weak_aug_images: torch.Tensor,
        strong_aug_images: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Args:
            weak_aug_images: Batch of weakly augmented drone images (B_u, C, H, W).
            strong_aug_images: Corresponding strongly augmented versions (B_u, C, H, W).

        Returns:
            loss: Unlabeled consistency cross-entropy loss.
            mask_ratio: Fraction of unlabeled samples exceeding confidence threshold.
        """
        # 1. Forward pass on weakly augmented images (detached to stop gradient)
        with torch.no_grad():
            weak_logits, _ = self.model(weak_aug_images)
            probs = F.softmax(weak_logits, dim=-1)
            max_probs, pseudo_labels = torch.max(probs, dim=-1)
            mask = max_probs.ge(self.confidence_threshold).float()

        if mask.sum() == 0:
            return torch.tensor(0.0, device=weak_aug_images.device, requires_grad=True), 0.0

        # 2. Forward pass on strongly augmented images
        strong_logits, _ = self.model(strong_aug_images)

        # 3. Compute consistency loss only on confident pseudo-labels
        loss_matrix = F.cross_entropy(strong_logits, pseudo_labels, reduction="none")
        unlabeled_loss = (loss_matrix * mask).mean() * self.unlabeled_weight
        mask_ratio = mask.mean().item()

        return unlabeled_loss, mask_ratio
