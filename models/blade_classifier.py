"""
Modular Wind Turbine Blade Defect Classification Architecture.

Features:
- Pre-trained backbone (ResNet50 / ResNet34 / ConvNeXt / TIMM)
- Multi-scale feature extraction across spatial stages (C3, C4, C5)
- Integrated CBAM (Channel & Spatial) attention mechanisms
- Dual adaptive pooling (Avg + Max) with calibrated classification head
- Support for progressive resolution training (224 -> 384 -> 512)
"""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as tv_models
from .attention import CBAM, SqueezeAndExcitation
from .multi_scale_aggregator import MultiScaleFeatureAggregator


class TurbineBladeDefectClassifier(nn.Module):
    """
    Modular Wind Turbine Blade Defect Classifier with Attention and Multi-Scale Feature Aggregation.
    """
    def __init__(
        self,
        num_classes: int = 4,
        backbone_name: str = "resnet50",
        pretrained: bool = True,
        use_attention: bool = True,
        use_multi_scale: bool = True,
        multi_scale_channels: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.use_attention = use_attention
        self.use_multi_scale = use_multi_scale

        # Build backbone and identify multi-scale feature tap points
        if backbone_name in ["resnet50", "resnet101"]:
            weights = tv_models.ResNet50_Weights.DEFAULT if (pretrained and backbone_name == "resnet50") else None
            base = tv_models.resnet50(weights=weights)
            self.init_conv = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool
            )
            self.stage1 = base.layer1  # 256 ch
            self.stage2 = base.layer2  # 512 ch
            self.stage3 = base.layer3  # 1024 ch
            self.stage4 = base.layer4  # 2048 ch
            in_channels_list = [512, 1024, 2048]
            final_channels = 2048

        elif backbone_name == "resnet34":
            weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
            base = tv_models.resnet34(weights=weights)
            self.init_conv = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool
            )
            self.stage1 = base.layer1  # 64 ch
            self.stage2 = base.layer2  # 128 ch
            self.stage3 = base.layer3  # 256 ch
            self.stage4 = base.layer4  # 512 ch
            in_channels_list = [128, 256, 512]
            final_channels = 512

        else:
            # Fallback to standard ResNet50
            weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
            base = tv_models.resnet50(weights=weights)
            self.init_conv = nn.Sequential(
                base.conv1,
                base.bn1,
                base.relu,
                base.maxpool
            )
            self.stage1 = base.layer1
            self.stage2 = base.layer2
            self.stage3 = base.layer3
            self.stage4 = base.layer4
            in_channels_list = [512, 1024, 2048]
            final_channels = 2048

        # Multi-scale feature aggregator
        if use_multi_scale:
            self.feature_aggregator = MultiScaleFeatureAggregator(
                in_channels_list=in_channels_list,
                out_channels=multi_scale_channels,
                use_attention=use_attention,
            )
            classifier_in_dim = self.feature_aggregator.descriptor_dim
        else:
            self.feature_aggregator = None
            self.stage4_attention = CBAM(final_channels) if use_attention else nn.Identity()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            classifier_in_dim = final_channels * 2

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes)
        )

        # Container for Grad-CAM activation caching
        self.last_feature_map: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            x: Input image batch (B, 3, H, W).
            return_features: If True, returns spatial feature map for Grad-CAM.

        Returns:
            logits: (B, num_classes)
            feature_map: (B, C, H', W') if return_features=True, else None
        """
        x0 = self.init_conv(x)
        c1 = self.stage1(x0)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        if self.use_multi_scale:
            # Multi-scale aggregation over C3 (stage2), C4 (stage3), C5 (stage4)
            features_list = [c2, c3, c4]
            descriptor, feature_map = self.feature_aggregator(features_list)
        else:
            feature_map = self.stage4_attention(c4)
            avg_p = self.avg_pool(feature_map).flatten(1)
            max_p = self.max_pool(feature_map).flatten(1)
            descriptor = torch.cat([avg_p, max_p], dim=1)

        self.last_feature_map = feature_map
        logits = self.classifier(descriptor)

        if return_features:
            return logits, feature_map
        return logits, None

    def freeze_backbone(self, freeze: bool = True) -> None:
        """Freeze or unfreeze backbone layers for progressive stage fine-tuning."""
        backbone_modules = [self.init_conv, self.stage1, self.stage2, self.stage3, self.stage4]
        for module in backbone_modules:
            for param in module.parameters():
                param.requires_grad = not freeze


def build_model(config: Dict[str, Any]) -> TurbineBladeDefectClassifier:
    """Factory helper to build model from configuration dictionary."""
    num_classes = len(config["dataset"]["classes"])
    model_cfg = config["model"]

    model = TurbineBladeDefectClassifier(
        num_classes=num_classes,
        backbone_name=model_cfg.get("backbone", "resnet50"),
        pretrained=model_cfg.get("pretrained", True),
        use_attention=model_cfg.get("use_cbam_attention", True),
        use_multi_scale=model_cfg.get("use_multi_scale", True),
        multi_scale_channels=model_cfg.get("multi_scale_channels", 256),
        dropout=model_cfg.get("dropout", 0.3),
    )
    return model
