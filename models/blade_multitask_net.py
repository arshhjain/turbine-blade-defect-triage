"""
Multi-Task Deep Convolutional Architecture for Wind Turbine Blade Defect & Severity Prediction.

Outputs simultaneously:
1. Defect Category (6 classes: crack, craze_crack, delamination, erosion, lightning_damage, surface_damage)
2. Damage Severity Level (Continuous scalar s in [1, 5] and Ordinal threshold probabilities P(S > k))

Includes:
- Pretrained ResNet50 backbone (timm or torchvision)
- Convolutional Block Attention Module (CBAM: Channel + Spatial attention)
- Concatenated Dual Adaptive Average & Max Pooling
- Calibrated Multi-Task classification & ordinal regression heads
"""

from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class ChannelAttention(nn.Module):
    """
    CBAM Channel Attention Module:
    Mc(F) = Sigmoid(MLP(AvgPool(F)) + MLP(MaxPool(F)))
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        reduced_channels = max(16, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale = self.sigmoid(avg_out + max_out)
        return x * scale


class SpatialAttention(nn.Module):
    """
    CBAM Spatial Attention Module:
    Ms(F) = Sigmoid(Conv7x7([AvgPool(F); MaxPool(F)]))
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        scale = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * scale


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module combining sequential Channel and Spatial Attention.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class TurbineBladeMultiTaskNet(nn.Module):
    """
    Multi-Task Neural Network for Wind Turbine Blade Anomaly Diagnosis.
    """
    def __init__(
        self,
        num_classes: int = 6,
        num_ordinal_thresholds: int = 4,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        use_cbam: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_ordinal_thresholds = num_ordinal_thresholds
        self.use_cbam = use_cbam
        
        # Load torchvision ResNet50 backbone
        if backbone_name == "resnet50":
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            base = tv_models.resnet50(weights=weights)
            in_features = 2048
        elif backbone_name == "resnet34":
            weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
            base = tv_models.resnet34(weights=weights)
            in_features = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1  # 256 ch (1/4 scale)
        self.layer2 = base.layer2  # 512 ch (1/8 scale)
        self.layer3 = base.layer3  # 1024 ch (1/16 scale)
        self.layer4 = base.layer4  # 2048 ch (1/32 scale)

        # Attention layer on deepest semantic feature map
        if self.use_cbam:
            self.cbam = CBAM(in_channels=in_features, reduction_ratio=16, kernel_size=7)
        else:
            self.cbam = nn.Identity()

        # Dual Adaptive Pooling: captures both salient peak activation and average context
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared representation projection with LayerNorm (stable across arbitrary batch sizes including B=1)
        fused_dim = in_features * 2
        self.shared_fc = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )

        # Head 1: Defect Category Classification (Logits in R^6)
        self.type_head = nn.Linear(512, num_classes)

        # Head 2a: Continuous Severity Regression (Scalar in [1.0, 5.0])
        self.severity_reg_head = nn.Linear(512, 1)

        # Head 2b: Ordinal Severity Classification (K-1 thresholds: P(S > 1), P(S > 2), P(S > 3), P(S > 4))
        self.ordinal_head = nn.Linear(512, num_ordinal_thresholds)

        # Save reference for Grad-CAM
        self.last_conv_layer = self.layer4

    def extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)
        
        attended_feat = self.cbam(feat_map)
        return feat_map, attended_feat

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, attended_feat = self.extract_features(x)

        avg_pooled = self.avg_pool(attended_feat).flatten(1)
        max_pooled = self.max_pool(attended_feat).flatten(1)
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)

        shared = self.shared_fc(pooled)

        # Defect category logits
        type_logits = self.type_head(shared)

        # Severity continuous prediction (scaled with sigmoid to range [1.0, 5.0])
        # s = 1.0 + 4.0 * Sigmoid(z)
        raw_sev = self.severity_reg_head(shared)
        severity_scalar = 1.0 + 4.0 * torch.sigmoid(raw_sev)

        # Ordinal classification logits for thresholds k in {1, 2, 3, 4}
        ordinal_logits = self.ordinal_head(shared)

        return {
            "type_logits": type_logits,
            "severity_scalar": severity_scalar,
            "ordinal_logits": ordinal_logits,
            "features": shared
        }

    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        """Inference helper returning argmax class and discrete ordinal severity."""
        out = self.forward(x)
        probs = F.softmax(out["type_logits"], dim=-1)
        pred_type = torch.argmax(probs, dim=-1)
        
        # Discrete ordinal severity from binary threshold activations:
        # S_discrete = 1 + sum_{k=1}^4 I(sigmoid(z_k) > 0.5)
        ord_probs = torch.sigmoid(out["ordinal_logits"])
        ord_discrete = 1 + torch.sum((ord_probs > 0.5).long(), dim=-1)
        
        return {
            "class_idx": pred_type,
            "class_probs": probs,
            "severity_continuous": out["severity_scalar"],
            "severity_discrete": ord_discrete,
            "ordinal_probs": ord_probs
        }
