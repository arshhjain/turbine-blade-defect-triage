"""
Unified Training Pipeline for Wind Turbine Blade Defect & Severity Diagnosis.

Supports:
1. Baseline B: Vanilla ResNet50 (Defect classification only, standard CE loss, no CBAM)
2. Proposed System: ResNet50 + CBAM Attention + Dual-Head Multi-Task (Type + Severity) + Class-Weighted Focal Loss
"""

import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import gc
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.blade_multitask_net import TurbineBladeMultiTaskNet
from datasets.multitask_dataset import TurbineBladeMultiTaskDataset, CLASSES, CLASS_TO_IDX
from datasets.augmentations import get_train_transforms, get_val_transforms
from training.multitask_losses import MultiTaskLoss, ClassWeightedFocalLoss


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    is_multitask: bool = True
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_type_loss = 0.0
    total_reg_loss = 0.0
    total_ord_loss = 0.0
    
    for images, targets_type, targets_sev, _ in loader:
        images = images.to(device)
        targets_type = targets_type.to(device)
        targets_sev = targets_sev.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        if is_multitask:
            loss_dict = criterion(
                type_logits=outputs["type_logits"],
                severity_scalar=outputs["severity_scalar"],
                ordinal_logits=outputs["ordinal_logits"],
                target_type=targets_type,
                target_severity=targets_sev
            )
            loss = loss_dict["total_loss"]
            total_type_loss += loss_dict["type_loss"].item()
            total_reg_loss += loss_dict["reg_loss"].item()
            total_ord_loss += loss_dict["ord_loss"].item()
        else:
            # Baseline single-task classification
            loss = criterion(outputs["type_logits"], targets_type)
            
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
    n_batches = max(1, len(loader))
    return {
        "loss": total_loss / n_batches,
        "type_loss": total_type_loss / n_batches if is_multitask else total_loss / n_batches,
        "reg_loss": total_reg_loss / n_batches if is_multitask else 0.0,
        "ord_loss": total_ord_loss / n_batches if is_multitask else 0.0
    }


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    is_multitask: bool = True
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    
    all_preds_type = []
    all_targets_type = []
    all_preds_sev = []
    all_targets_sev = []
    
    for images, targets_type, targets_sev, _ in loader:
        images = images.to(device)
        targets_type = targets_type.to(device)
        targets_sev = targets_sev.to(device)
        
        outputs = model(images)
        
        if is_multitask:
            loss_dict = criterion(
                type_logits=outputs["type_logits"],
                severity_scalar=outputs["severity_scalar"],
                ordinal_logits=outputs["ordinal_logits"],
                target_type=targets_type,
                target_severity=targets_sev
            )
            total_loss += loss_dict["total_loss"].item()
            
            # Severity prediction
            ord_probs = torch.sigmoid(outputs["ordinal_logits"])
            ord_discrete = 1 + torch.sum((ord_probs > 0.5).long(), dim=-1)
            all_preds_sev.extend(ord_discrete.cpu().numpy())
        else:
            loss = criterion(outputs["type_logits"], targets_type)
            total_loss += loss.item()
            
        preds_type = torch.argmax(outputs["type_logits"], dim=1)
        all_preds_type.extend(preds_type.cpu().numpy())
        all_targets_type.extend(targets_type.cpu().numpy())
        all_targets_sev.extend(targets_sev.cpu().numpy())
        
    n_batches = max(1, len(loader))
    acc = accuracy_score(all_targets_type, all_preds_type)
    macro_f1 = f1_score(all_targets_type, all_preds_type, average="macro", zero_division=0)
    
    sev_mae = mean_absolute_error(all_targets_sev, all_preds_sev) if is_multitask else None
    
    return {
        "val_loss": total_loss / n_batches,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "sev_mae": sev_mae,
        "targets_type": all_targets_type,
        "preds_type": all_preds_type,
        "targets_sev": all_targets_sev,
        "preds_sev": all_preds_sev
    }


def run_training(
    run_name: str = "proposed_multitask_cbam",
    is_multitask: bool = True,
    use_cbam: bool = True,
    use_focal: bool = True,
    image_size: int = 384,
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 0.0003,
    weight_decay: float = 0.0001,
    seed: int = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"RUN: {run_name} | Device: {device} | Multitask: {is_multitask} | CBAM: {use_cbam}")
    print(f"Resolution: {image_size}x{image_size} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print(f"========================================================\n")
    
    # Datasets
    train_transform = get_train_transforms(image_size=image_size, heavy=True)
    val_transform = get_val_transforms(image_size=image_size)
    
    train_dataset = TurbineBladeMultiTaskDataset(split="train", transform=train_transform, image_size=image_size)
    val_dataset = TurbineBladeMultiTaskDataset(split="val", transform=val_transform, image_size=image_size)
    test_dataset = TurbineBladeMultiTaskDataset(split="test", transform=val_transform, image_size=image_size)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # Model
    model = TurbineBladeMultiTaskNet(
        num_classes=len(CLASSES),
        num_ordinal_thresholds=4,
        backbone_name="resnet50",
        pretrained=True,
        use_cbam=use_cbam,
        dropout=0.3
    ).to(device)
    
    # Loss function
    if is_multitask:
        class_weights = train_dataset.get_class_weights().to(device) if use_focal else None
        criterion = MultiTaskLoss(
            class_weights=class_weights,
            focal_gamma=2.0 if use_focal else 0.0,
            lambda_reg=0.5,
            lambda_ord=0.5
        )
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    out_dir = PROJECT_ROOT / "outputs" / run_name
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_f1 = -1.0
    history = []
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        t_epoch_0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, is_multitask=is_multitask)
        val_metrics = evaluate_epoch(model, val_loader, criterion, device, is_multitask=is_multitask)
        scheduler.step()
        epoch_dur = time.time() - t_epoch_0
        
        entry = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["val_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_sev_mae": val_metrics["sev_mae"],
            "lr": optimizer.param_groups[0]["lr"],
            "duration_sec": epoch_dur
        }
        history.append(entry)
        
        sev_log = f" | Val Sev MAE: {val_metrics['sev_mae']:.3f}" if is_multitask else ""
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['val_loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']*100:.2f}% | Val F1: {val_metrics['macro_f1']*100:.2f}%"
              f"{sev_log} | {epoch_dur:.1f}s", flush=True)
        gc.collect()
        
        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": {
                "run_name": run_name,
                "is_multitask": is_multitask,
                "use_cbam": use_cbam,
                "image_size": image_size,
                "classes": CLASSES
            }
        }, ckpt_dir / "last.pth")
        
        # Save best checkpoint based on Val Macro F1
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_sev_mae": val_metrics["sev_mae"],
                "config": {
                    "run_name": run_name,
                    "is_multitask": is_multitask,
                    "use_cbam": use_cbam,
                    "image_size": image_size,
                    "classes": CLASSES
                }
            }, ckpt_dir / "best.pth")
            
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time/60:.2f} minutes. Best Val F1: {best_val_f1*100:.2f}%.")
    
    # Final Held-Out Test Split Evaluation
    print("\n" + "=" * 65)
    print(f"EVALUATING BEST CHECKPOINT ON HELD-OUT TEST SPLIT (N={len(test_dataset)})")
    print("=" * 65)
    best_ckpt = torch.load(ckpt_dir / "best.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_eval = evaluate_epoch(model, test_loader, criterion, device, is_multitask=is_multitask)
    
    test_rep_dict = classification_report(test_eval["targets_type"], test_eval["preds_type"], target_names=CLASSES, output_dict=True, zero_division=0)
    test_rep_str = classification_report(test_eval["targets_type"], test_eval["preds_type"], target_names=CLASSES, zero_division=0)
    test_cm = confusion_matrix(test_eval["targets_type"], test_eval["preds_type"])
    
    print(f"Test Accuracy : {test_eval['accuracy']*100:.2f}% | Test Macro F1: {test_eval['macro_f1']*100:.2f}%")
    if is_multitask:
        test_sev_cm = confusion_matrix(test_eval["targets_sev"], test_eval["preds_sev"], labels=[1, 2, 3, 4, 5])
        print(f"Severity MAE  : {test_eval['sev_mae']:.3f} levels")
    print("\nDefect Classification Breakdown:")
    print(test_rep_str)
    print("Confusion Matrix:")
    print(test_cm)
    
    # Save test metrics and training history
    results = {
        "run_name": run_name,
        "is_multitask": is_multitask,
        "use_cbam": use_cbam,
        "image_size": image_size,
        "epochs": epochs,
        "total_train_time_sec": total_time,
        "best_epoch": best_ckpt["epoch"],
        "test_accuracy": float(test_eval["accuracy"]),
        "test_macro_f1": float(test_eval["macro_f1"]),
        "test_sev_mae": float(test_eval["sev_mae"]) if is_multitask else None,
        "classification_report": test_rep_dict,
        "confusion_matrix": test_cm.tolist(),
        "severity_confusion_matrix": test_sev_cm.tolist() if is_multitask else None,
        "history": history
    }
    
    with open(out_dir / "test_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\nFinal evaluation saved to: {out_dir / 'test_evaluation_report.json'}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="proposed_multitask_cbam")
    parser.add_argument("--multitask", action="store_true", default=True)
    parser.add_argument("--single-task", dest="multitask", action="store_false")
    parser.add_argument("--cbam", action="store_true", default=True)
    parser.add_argument("--no-cbam", dest="cbam", action="store_false")
    parser.add_argument("--focal", action="store_true", default=True)
    parser.add_argument("--no-focal", dest="focal", action="store_false")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_training(
        run_name=args.run_name,
        is_multitask=args.multitask,
        use_cbam=args.cbam,
        use_focal=args.focal,
        image_size=args.image_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed
    )
