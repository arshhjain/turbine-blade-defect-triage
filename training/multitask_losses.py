"""
Loss Formulations for Wind Turbine Blade Multi-Task Defect & Severity Learning.

Addresses:
- Safety-critical recall collapse via Class-Weighted Focal Cross-Entropy
- Natural ordinal distance penalization via Smooth L1 & Ordinal Binary Logistic Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class ClassWeightedFocalLoss(nn.Module):
    """
    Multi-Class Focal Loss with inverse class weighting:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        inputs: (B, C) unnormalized logits
        targets: (B,) class indices
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p = torch.exp(-ce_loss)  # p_t
        focal_weight = (1.0 - p) ** self.gamma
        
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_weight = self.alpha[targets]
            focal_loss = alpha_weight * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class MultiTaskLoss(nn.Module):
    """
    Joint loss function combining:
    1. Defect Category Classification (Class-Weighted Focal Loss)
    2. Continuous Severity Regression (Smooth L1 Loss)
    3. Ordinal Severity Threshold Classification (Frank-Hall BCE Formulation)
    """
    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        focal_gamma: float = 2.0,
        lambda_reg: float = 0.5,
        lambda_ord: float = 0.5
    ):
        super().__init__()
        self.type_loss_fn = ClassWeightedFocalLoss(alpha=class_weights, gamma=focal_gamma)
        self.reg_loss_fn = nn.SmoothL1Loss(beta=0.5)
        self.bce_loss_fn = nn.BCEWithLogitsLoss()
        self.lambda_reg = lambda_reg
        self.lambda_ord = lambda_ord

    def forward(
        self,
        type_logits: torch.Tensor,
        severity_scalar: torch.Tensor,
        ordinal_logits: torch.Tensor,
        target_type: torch.Tensor,
        target_severity: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        type_logits: (B, C)
        severity_scalar: (B, 1) in [1.0, 5.0]
        ordinal_logits: (B, 4) logits for thresholds k in {1, 2, 3, 4}
        target_type: (B,) in {0, ..., C-1}
        target_severity: (B,) in {1, ..., 5} float
        """
        # 1. Defect type focal loss
        l_type = self.type_loss_fn(type_logits, target_type)

        # 2. Continuous severity regression loss
        l_reg = self.reg_loss_fn(severity_scalar.squeeze(-1), target_severity.float())

        # 3. Ordinal binary threshold loss (Frank-Hall method)
        # For thresholds k in [1, 2, 3, 4], target is 1 if severity > k, else 0
        thresholds = torch.tensor([1, 2, 3, 4], device=target_severity.device).unsqueeze(0)  # (1, 4)
        target_expanded = target_severity.unsqueeze(-1)  # (B, 1)
        ord_binary_targets = (target_expanded > thresholds).float()  # (B, 4)
        l_ord = self.bce_loss_fn(ordinal_logits, ord_binary_targets)

        # Total combined loss
        total_loss = l_type + self.lambda_reg * l_reg + self.lambda_ord * l_ord

        return {
            "total_loss": total_loss,
            "type_loss": l_type,
            "reg_loss": l_reg,
            "ord_loss": l_ord
        }
