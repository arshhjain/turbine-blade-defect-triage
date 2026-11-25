"""
Multi-Scale Feature Processing Module for Wind Turbine Blade Defect Classification.

Captures extreme scale variance across defect categories:
- Micro-cracks & hairline fractures: High-frequency spatial gradients in shallow/mid-level features.
- Leading-edge erosion: Intermediate-scale cluster patterns and texture roughness.
- Delamination: Wide-area structural composite disbonding in deep semantic features.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import CBAM


class MultiScaleFeatureAggregator(nn.Module):
    """
    Multi-Scale Feature Aggregator that fuses multi-resolution feature maps
    (e.g., C3, C4, C5 from ResNet stages) using lateral projections and CBAM attention.
    """
    def __init__(
        self,
        in_channels_list: List[int] = [512, 1024, 2048],
        out_channels: int = 256,
        use_attention: bool = True,
    ):
        super().__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels
        self.use_attention = use_attention

        # Lateral 1x1 convolutions to project all stages to uniform channel dimension
        self.laterals = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for in_ch in in_channels_list
        ])

        # Smooth convolutions for each scale
        self.smooth_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for _ in in_channels_list
        ])

        # Attention blocks for each scale to suppress background noise
        if use_attention:
            self.attention_blocks = nn.ModuleList([
                CBAM(out_channels) for _ in in_channels_list
            ])
        else:
            self.attention_blocks = None

        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(out_channels * len(in_channels_list), out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Global multi-scale attention
        self.global_cbam = CBAM(out_channels) if use_attention else nn.Identity()

        # Dual pooling: adaptive avg + max pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Total descriptor dimension
        self.descriptor_dim = out_channels * 2

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: List of feature tensors from backbone stages [C3, C4, C5].

        Returns:
            fused_descriptor: 1D feature vector (B, descriptor_dim).
            target_feature_map: Spatial 2D feature map for Grad-CAM explainability (B, out_channels, H, W).
        """
        assert len(features) == len(self.in_channels_list), \
            f"Expected {len(self.in_channels_list)} feature maps, got {len(features)}"

        # 1. Project through lateral convolutions
        proj_features = [lateral(f) for lateral, f in zip(self.laterals, features)]

        # 2. Align spatial dimensions to C4 resolution (intermediate scale)
        target_h, target_w = proj_features[1].shape[2], proj_features[1].shape[3]
        aligned_features = []

        for i, f in enumerate(proj_features):
            if f.shape[2:] != (target_h, target_w):
                f_aligned = F.interpolate(f, size=(target_h, target_w), mode="bilinear", align_corners=False)
            else:
                f_aligned = f

            # Apply smooth conv
            f_smoothed = self.smooth_convs[i](f_aligned)

            # Apply scale-specific attention
            if self.attention_blocks is not None:
                f_smoothed = self.attention_blocks[i](f_smoothed)

            aligned_features.append(f_smoothed)

        # 3. Concatenate and fuse multi-scale features
        cat_features = torch.cat(aligned_features, dim=1)
        fused_map = self.fusion_conv(cat_features)
        fused_map = self.global_cbam(fused_map)

        # 4. Dual pooling (concatenated avg + max pool) to preserve both mean texture and peak activations
        avg_vec = self.avg_pool(fused_map).flatten(1)
        max_vec = self.max_pool(fused_map).flatten(1)
        fused_descriptor = torch.cat([avg_vec, max_vec], dim=1)

        return fused_descriptor, fused_map
