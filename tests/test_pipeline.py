"""
Comprehensive Integration Test Suite for Wind Turbine Blade Defect Classification MVP.
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.mock_generator import (
    draw_blade_background,
    inject_cracking_defect,
    inject_erosion_defect,
    inject_delamination_defect,
    generate_synthetic_turbine_dataset,
)
from datasets.augmentations import get_train_transforms, get_val_transforms
from models.attention import CBAM, ChannelAttention, SpatialAttention
from models.multi_scale_aggregator import MultiScaleFeatureAggregator
from models.blade_classifier import TurbineBladeDefectClassifier
from training.losses import FocalLoss, LabelSmoothingCrossEntropy
from inference.gradcam import GradCAM
from inference.diagnostic_engine import BladeDiagnosticEngine


class TestWindTurbinePipeline(unittest.TestCase):
    """Integration test suite covering full ML pipeline."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["normal", "cracking", "erosion", "delamination"]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def test_01_procedural_synthetic_generator(self):
        """Test procedural generator produces non-empty RGB images and defect bounding boxes."""
        base_img = draw_blade_background(256, 256)
        self.assertEqual(base_img.shape, (256, 256, 3))
        self.assertEqual(base_img.dtype, np.uint8)

        crack_img, crack_box = inject_cracking_defect(base_img.copy())
        self.assertEqual(crack_img.shape, (256, 256, 3))
        self.assertEqual(len(crack_box), 4)

        erosion_img, erosion_box = inject_erosion_defect(base_img.copy())
        self.assertEqual(erosion_img.shape, (256, 256, 3))
        self.assertEqual(len(erosion_box), 4)

        delam_img, delam_box = inject_delamination_defect(base_img.copy())
        self.assertEqual(delam_img.shape, (256, 256, 3))
        self.assertEqual(len(delam_box), 4)

    def test_02_albumentations_pipeline(self):
        """Test Albumentations data augmentation and progressive resizing."""
        dummy_img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)

        for res in [224, 384, 512]:
            train_t = get_train_transforms(image_size=res, heavy=True)
            val_t = get_val_transforms(image_size=res)

            out_train = train_t(image=dummy_img)["image"]
            out_val = val_t(image=dummy_img)["image"]

            self.assertEqual(out_train.shape, (3, res, res))
            self.assertEqual(out_val.shape, (3, res, res))

    def test_03_attention_modules(self):
        """Test ChannelAttention, SpatialAttention, and CBAM modules."""
        x = torch.randn(2, 64, 28, 28)
        ca = ChannelAttention(in_planes=64)
        sa = SpatialAttention()
        cbam = CBAM(in_planes=64)

        out_ca = ca(x)
        self.assertEqual(out_ca.shape, (2, 64, 1, 1))

        out_sa = sa(x)
        self.assertEqual(out_sa.shape, (2, 1, 28, 28))

        out_cbam = cbam(x)
        self.assertEqual(out_cbam.shape, (2, 64, 28, 28))

    def test_04_multi_scale_aggregator(self):
        """Test multi-scale feature aggregator fuses C3, C4, C5."""
        c3 = torch.randn(2, 512, 28, 28)
        c4 = torch.randn(2, 1024, 14, 14)
        c5 = torch.randn(2, 2048, 7, 7)

        agg = MultiScaleFeatureAggregator(
            in_channels_list=[512, 1024, 2048],
            out_channels=256,
            use_attention=True,
        )
        desc, fmap = agg([c3, c4, c5])

        self.assertEqual(desc.shape, (2, 512))  # 256 avg + 256 max
        self.assertEqual(fmap.shape, (2, 256, 14, 14))

    def test_05_model_forward_and_backward(self):
        """Test full model forward pass and backward gradient flow."""
        model = TurbineBladeDefectClassifier(
            num_classes=4,
            backbone_name="resnet50",
            pretrained=False,
            use_attention=True,
            use_multi_scale=True,
        ).to(self.device)

        inputs = torch.randn(2, 3, 224, 224, device=self.device)
        targets = torch.tensor([1, 2], device=self.device)

        logits, fmap = model(inputs, return_features=True)
        self.assertEqual(logits.shape, (2, 4))
        self.assertIsNotNone(fmap)

        loss_fn = FocalLoss(gamma=2.0)
        loss = loss_fn(logits, targets)
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(model.classifier[-1].weight.grad)

    def test_06_gradcam_and_diagnostics(self):
        """Test Grad-CAM heatmap generation and structured diagnostic report."""
        model = TurbineBladeDefectClassifier(
            num_classes=4,
            backbone_name="resnet50",
            pretrained=False,
            use_attention=True,
            use_multi_scale=True,
        ).to(self.device)

        test_dir = PROJECT_ROOT / "outputs/test_scratch"
        test_dir.mkdir(parents=True, exist_ok=True)
        img_path = test_dir / "test_blade.jpg"

        # Create dummy image on disk
        dummy = draw_blade_background(256, 256)
        import cv2
        cv2.imwrite(str(img_path), dummy)

        engine = BladeDiagnosticEngine(
            model=model,
            class_to_idx=self.class_to_idx,
            device=self.device,
            image_size=256,
            output_dir=str(test_dir),
        )

        record = engine.diagnose_image(str(img_path), generate_cam=True)

        self.assertIn("primary_defect", record)
        self.assertIn("severity_level", record)
        self.assertIn("actionable_protocol", record)
        self.assertIn("cam_overlay_saved_at", record)
        self.assertTrue(Path(record["cam_overlay_saved_at"]).exists())


if __name__ == "__main__":
    unittest.main()
