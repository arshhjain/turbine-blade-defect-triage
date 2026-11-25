"""
Progressive Training CLI for Wind Turbine Blade Defect Classification.

Executes resolution progression training strategy:
- Low-resolution stage (224x224): Learn coarse blade contours & macro features
- Mid-resolution stage (384x384): Intermediate erosion pitting and laminate boundaries
- High-resolution stage (512x512): Fine-tune on intricate hairline cracks and micro-defects
"""

import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.blade_classifier import TurbineBladeDefectClassifier, build_model
from training.progressive_trainer import ProgressiveResolutionTrainer
from datasets.unified_loader import (
    parse_figshare_dataset,
    parse_roboflow_yolo_dataset,
    parse_generic_directory,
    CANONICAL_CLASS_MAP,
)


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def gather_all_samples(config: dict):
    target_classes = config["dataset"]["classes"]
    class_to_idx = {cls: idx for idx, cls in enumerate(target_classes)}
    all_samples = []

    # 1. Figshare
    figshare_dir = PROJECT_ROOT / config["dataset"].get("figshare_dir", "data/figshare")
    all_samples.extend(parse_figshare_dataset(str(figshare_dir), target_classes))

    # 2. Roboflow
    roboflow_dir = PROJECT_ROOT / config["dataset"].get("roboflow_dir", "data/roboflow")
    all_samples.extend(parse_roboflow_yolo_dataset(str(roboflow_dir), target_classes))

    # Deduplicate
    unique_samples = []
    seen = set()
    for s in all_samples:
        key = (s.image_path, s.label, str(s.bbox))
        if key not in seen:
            seen.add(key)
            unique_samples.append(s)

    return unique_samples, class_to_idx


def main():
    parser = argparse.ArgumentParser(description="Train Wind Turbine Blade Defect Classification Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--epochs-per-stage", type=int, default=None, help="Override epochs per stage for fast testing")
    parser.add_argument("--device", type=str, default=None, help="Force device ('cuda' or 'cpu')")
    parser.add_argument("--fast-dev-run", action="store_true", help="Quick 1-epoch test per stage")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    config = load_config(str(config_path))

    # Set device
    if args.device:
        device_str = args.device
    else:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # Set random seeds
    seed = config["project"].get("random_seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Fast dev run adjustments
    if args.fast_dev_run:
        for stage in config["progressive_training"]["stages"]:
            stage["epochs"] = 1
            stage["batch_size"] = min(stage["batch_size"], 4)
    elif args.epochs_per_stage is not None:
        for stage in config["progressive_training"]["stages"]:
            stage["epochs"] = args.epochs_per_stage

    # Gather data samples
    samples, class_to_idx = gather_all_samples(config)
    print(f"Discovered {len(samples)} wind turbine blade inspection samples across categories: {list(class_to_idx.keys())}")

    if len(samples) == 0:
        raise RuntimeError(f"No wind turbine inspection images discovered in {figshare_dir}. Please ensure dataset is unpacked.")

    # Split train and validation
    labels = [s.label for s in samples]
    val_split = config["dataset"].get("val_split", 0.2)
    train_samples, val_samples = train_test_split(
        samples,
        test_size=val_split,
        random_state=seed,
        stratify=labels if len(set(labels)) > 1 else None
    )

    print(f"Training set: {len(train_samples)} samples | Validation set: {len(val_samples)} samples")

    # Build model
    model = build_model(config).to(device)
    print(f"Built TurbineBladeDefectClassifier: Backbone={config['model']['backbone']}, CBAM Attention={config['model']['use_cbam_attention']}, Multi-Scale={config['model']['use_multi_scale']}")

    # Progressive Trainer
    trainer = ProgressiveResolutionTrainer(
        model=model,
        train_samples=train_samples,
        val_samples=val_samples,
        class_to_idx=class_to_idx,
        config=config,
        device=device,
    )

    results = trainer.run_progressive_training()
    print("\nTraining summary:")
    print(f"Peak Validation F1-Score: {results['best_val_f1'] * 100:.2f}%")
    print(f"Model Checkpoint: {results['best_checkpoint']}")


if __name__ == "__main__":
    main()
