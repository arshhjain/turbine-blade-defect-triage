"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Drone Blade Defect Localization.

Provides visual explainability for RGB drone images, overlaying thermal heatmaps
directly highlighting the detected crack lines, erosion patches, or delamination blisters.
"""

from typing import Optional, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM implementation hooked into multi-scale feature aggregator or backbone stages.
    """
    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()

        # Identify target layer if not explicitly provided
        if target_layer is None:
            if hasattr(model, "feature_aggregator") and model.feature_aggregator is not None:
                self.target_layer = model.feature_aggregator.fusion_conv
            elif hasattr(model, "stage4"):
                self.target_layer = model.stage4
            else:
                raise ValueError("Could not automatically locate target layer for GradCAM.")
        else:
            self.target_layer = target_layer

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            # Output can be a tensor or a tuple
            if isinstance(output, tuple):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate spatial Grad-CAM heatmap for the given input tensor.

        Args:
            input_tensor: (1, 3, H, W) normalized tensor.
            target_class_idx: If None, uses top predicted class.

        Returns:
            heatmap: (H, W) float32 array in [0, 1].
            predicted_class: Target class index used.
            confidence: Softmax probability of target class.
        """
        self.model.zero_grad()
        logits, _ = self.model(input_tensor, return_features=True)
        probs = F.softmax(logits, dim=-1)

        if target_class_idx is None:
            target_class_idx = int(torch.argmax(logits, dim=1).item())

        confidence = float(probs[0, target_class_idx].item())

        # Backward pass for the target class logit
        target_score = logits[0, target_class_idx]
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients/activations.")

        # Global average pooling of gradients over spatial dimensions (H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # Only positive influence

        # Resize CAM to match input tensor resolution
        cam = F.interpolate(
            cam,
            size=(input_tensor.shape[2], input_tensor.shape[3]),
            mode="bilinear",
            align_corners=False
        )

        cam_np = cam.squeeze().cpu().numpy()
        # Normalize to [0, 1]
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-8:
            heatmap = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            heatmap = np.zeros_like(cam_np)

        return heatmap, target_class_idx, confidence

    def overlay_heatmap(
        self,
        rgb_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay colored heatmap on original RGB blade image.

        Args:
            rgb_image: (H, W, 3) uint8 image.
            heatmap: (H, W) float32 array [0, 1].
            alpha: Heatmap blend factor.
            colormap: OpenCV colormap.

        Returns:
            blended: (H, W, 3) RGB uint8 image with overlay.
        """
        h, w, _ = rgb_image.shape
        heatmap_resized = cv2.resize(heatmap, (w, h))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colored_cam = cv2.applyColorMap(heatmap_uint8, colormap)
        colored_cam_rgb = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)

        blended = np.clip(
            alpha * colored_cam_rgb.astype(np.float32) + (1.0 - alpha) * rgb_image.astype(np.float32),
            0,
            255
        ).astype(np.uint8)

        return blended
