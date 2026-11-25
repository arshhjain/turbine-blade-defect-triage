"""
Evaluation Script for Wind Turbine Blade Defect Classification MVP.

Computes:
- Per-class precision, recall, and F1-scores
- Overall macro-F1 and top-1 accuracy
- Confusion matrix and classification report
"""

import os
import sys
import argparse
import json
from pathlib import Path
import yaml
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.blade_classifier import TurbineBladeDefectClassifier, build_model
from datasets.augmentations import get_val_transforms
from datasets.unified_loader import BladeDefectDataset, parse_generic_directory


def main():
    parser = argparse.ArgumentParser(description="Evaluate Wind Turbine Blade Defect Classifier")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model_progressive.pth", help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--test-dir", type=str, default=None, help="Optional test directory")
    parser.add_argument("--resolution", type=int, default=512, help="Evaluation image resolution")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    args = parser.parse_args()

    # Load checkpoint
    ckpt_path = PROJECT_ROOT / args.checkpoint
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}. Please train model first.")

    device_str = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    checkpoint = torch.load(ckpt_path, map_location=device)
    config = checkpoint.get("config", {})
    if not config:
        with open(PROJECT_ROOT / args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    class_to_idx = checkpoint.get("class_to_idx", {cls: i for i, cls in enumerate(config["dataset"]["classes"])})
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Build model and load weights
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Data to evaluate
    target_classes = list(class_to_idx.keys())
    test_samples = []

    # Check if test_dir is provided or fallback to config
    test_dir_path = PROJECT_ROOT / args.test_dir if args.test_dir else PROJECT_ROOT / config["dataset"].get("figshare_dir", "data/figshare")
    
    # Try parsing via Figshare VOC parser first, then generic directory
    from datasets.unified_loader import parse_figshare_dataset
    test_samples = parse_figshare_dataset(str(test_dir_path), target_classes)
    if not test_samples:
        test_samples = parse_generic_directory(str(test_dir_path), target_classes)

    if not test_samples:
        raise ValueError(f"No evaluation samples found in {test_dir}")

    val_transform = get_val_transforms(image_size=args.resolution)
    dataset = BladeDefectDataset(test_samples, class_to_idx, transform=val_transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    all_preds = []
    all_targets = []

    print(f"Evaluating {len(dataset)} drone blade images at {args.resolution}x{args.resolution} resolution...")
    with torch.no_grad():
        for images, targets, _ in loader:
            images = images.to(device)
            logits, _ = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    report_dict = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    report_str = classification_report(all_targets, all_preds, target_names=class_names, zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    print("\n" + "=" * 60)
    print("WIND TURBINE BLADE DEFECT CLASSIFICATION REPORT")
    print("=" * 60)
    print(f"Overall Accuracy: {acc * 100:.2f}% | Macro F1-Score: {macro_f1 * 100:.2f}%\n")
    print(report_str)
    print("Confusion Matrix:")
    print(cm)
    print("=" * 60)

    # Save evaluation metrics
    eval_out_dir = PROJECT_ROOT / "outputs/evaluations"
    eval_out_dir.mkdir(parents=True, exist_ok=True)
    results_path = eval_out_dir / "evaluation_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": acc,
            "macro_f1": macro_f1,
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
        }, f, indent=2)
    print(f"Evaluation report saved to: {results_path}")


if __name__ == "__main__":
    main()
