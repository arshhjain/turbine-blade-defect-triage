"""
Attention Mechanisms for Wind Turbine Blade Defect Classification.

Implements:
1. Channel Attention Module (CAM)
2. Spatial Attention Module (SAM)
3. Convolutional Block Attention Module (CBAM)
4. Squeeze-and-Excitation (SE) Block

These mechanisms isolate blade surface anomalies (cracks, erosion, delamination)
from complex background textures (sky, clouds, horizon, tower shadows, and ocean glare).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) for isolating defect feature channels.
    Combines both AvgPool and MaxPool descriptors with a shared multi-layer perceptron.
    """
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        reduced_planes = max(in_planes // ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, reduced_planes, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) for isolating anomaly locations on the blade surface
    while suppressing extraneous background textures (sky, clouds, terrain).
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(combined)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module combining sequential Channel and Spatial Attention.
    """
    def __init__(self, in_planes: int, ratio: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out


class SqueezeAndExcitation(nn.Module):
    """
    Squeeze-and-Excitation (SE) Block for adaptive channel-wise recalibration.
    """
    def __init__(self, in_planes: int, reduction: int = 16):
        super().__init__()
        reduced = max(in_planes // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_planes, reduced, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, in_planes, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.fc(x)
        return x * weight
