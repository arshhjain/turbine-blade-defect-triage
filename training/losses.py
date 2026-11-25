"""
Loss Functions for Wind Turbine Blade Defect Classification.

Implements:
1. Multi-Class Focal Loss (handles severe class imbalance in drone inspection imagery)
2. Label Smoothing Cross Entropy Loss
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss to address severe class imbalance in drone blade inspection.
    Downweights easy background/normal examples and forces gradient focus on
    hard-to-detect hairline cracks and subtle delamination boundaries.
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits from model (B, C).
            targets: Target class indices (B).
        """
        log_pt = F.log_softmax(inputs, dim=-1)
        pt = torch.exp(log_pt)

        # Gather probability of true target
        log_pt = log_pt.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = pt.gather(1, targets.unsqueeze(1)).squeeze(1)

        focal_weight = torch.pow((1.0 - pt), self.gamma)

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha.gather(0, targets)
            loss = -alpha_t * focal_weight * log_pt
        else:
            loss = -focal_weight * log_pt

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing to prevent model overconfidence
    on noisy drone annotations.
    """
    def __init__(self, smoothing: float = 0.05, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(inputs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        loss = (-true_dist * log_probs).sum(dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_loss(config: dict, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """Build loss function based on configuration."""
    loss_type = config["training"].get("loss_type", "focal").lower()
    if loss_type == "focal":
        gamma = config["training"].get("focal_gamma", 2.0)
        return FocalLoss(gamma=gamma, alpha=class_weights)
    elif loss_type == "label_smoothing":
        smoothing = config["training"].get("label_smoothing", 0.05)
        return LabelSmoothingCrossEntropy(smoothing=smoothing)
    else:
        return nn.CrossEntropyLoss(weight=class_weights)
