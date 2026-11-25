"""
Resolution Progression Training Engine for Wind Turbine Blade Defect Classification.

Strategy:
- Stage 1 (224x224): Rapid learning of coarse blade silhouettes, edge boundaries, and macro-flaws.
- Stage 2 (384x384): Intermediate learning of leading-edge erosion pitting and composite delamination.
- Stage 3 (512x512): High-resolution fine-tuning on subtle hairline cracks and intricate fiber fractures.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

from datasets.augmentations import get_train_transforms, get_val_transforms
from datasets.unified_loader import BladeDefectDataset
from .losses import build_loss


class ProgressiveResolutionTrainer:
    """
    Manages multi-stage progressive resolution training across epochs.
    """
    def __init__(
        self,
        model: nn.Module,
        train_samples: list,
        val_samples: list,
        class_to_idx: Dict[str, int],
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.model = model
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.config = config
        self.device = device

        raw_save_dir = Path(config["training"].get("save_dir", "outputs/checkpoints"))
        if not raw_save_dir.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            self.save_dir = project_root / raw_save_dir
        else:
            self.save_dir = raw_save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.save_dir / "best_model_progressive.pth"

        # Loss function
        self.criterion = build_loss(config).to(device)

        # Tracking metrics
        self.history = []
        self.best_val_f1 = -1.0
        self.current_epoch = 0

    def create_loaders_for_stage(
        self,
        resolution: int,
        batch_size: int,
    ) -> Tuple[DataLoader, DataLoader]:
        """Dynamically create stage-specific dataloaders with updated Albumentations resolution."""
        heavy = self.config["augmentation"].get("heavy", True)
        train_transform = get_train_transforms(image_size=resolution, heavy=heavy)
        val_transform = get_val_transforms(image_size=resolution)

        train_ds = BladeDefectDataset(
            self.train_samples, self.class_to_idx, transform=train_transform
        )
        val_ds = BladeDefectDataset(
            self.val_samples, self.class_to_idx, transform=val_transform
        )

        num_workers = self.config["dataset"].get("num_workers", 0)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False
        )

        return train_loader, val_loader

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float, float]:
        """Train one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        for images, targets, _ in train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            optimizer.zero_grad()
            logits, _ = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()

            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.detach().cpu().numpy())

        total = len(train_loader.dataset)
        epoch_loss = running_loss / max(total, 1)
        epoch_acc = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0
        epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) if len(all_targets) > 0 else 0.0

        return epoch_loss, epoch_acc, epoch_f1

    def evaluate(
        self,
        val_loader: DataLoader,
    ) -> Tuple[float, float, float, float, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                logits, _ = self.model(images)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * images.size(0)
                preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(targets.detach().cpu().numpy())

        total = len(val_loader.dataset)
        val_loss = running_loss / max(total, 1)
        val_acc = accuracy_score(all_targets, all_preds) if len(all_targets) > 0 else 0.0
        val_precision = precision_score(all_targets, all_preds, average="macro", zero_division=0) if len(all_targets) > 0 else 0.0
        val_recall = recall_score(all_targets, all_preds, average="macro", zero_division=0) if len(all_targets) > 0 else 0.0
        val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) if len(all_targets) > 0 else 0.0

        return val_loss, val_acc, val_precision, val_recall, val_f1

    def run_progressive_training(self) -> Dict[str, Any]:
        """
        Execute full multi-stage progressive resolution training pipeline.
        """
        stages = self.config["progressive_training"]["stages"]
        print("=" * 70)
        print("STARTING WIND TURBINE DEFECT RESOLUTION PROGRESSION TRAINING")
        print(f"Total Stages: {len(stages)} | Device: {self.device}")
        print("=" * 70)

        for stage_idx, stage in enumerate(stages):
            stage_name = stage.get("name", f"stage_{stage_idx+1}")
            res = stage["resolution"]
            epochs = stage["epochs"]
            batch_size = stage["batch_size"]
            lr = stage["lr"]
            freeze_epochs = stage.get("freeze_backbone_epochs", 0)

            print(f"\n--- [Stage {stage_idx + 1}/{len(stages)}] {stage_name.upper()} | Resolution: {res}x{res} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr} ---")

            # Dynamic dataloaders for new resolution
            train_loader, val_loader = self.create_loaders_for_stage(res, batch_size)

            # Stage-specific optimizer and scheduler
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=lr,
                weight_decay=stage.get("weight_decay", 1e-4)
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=self.config["training"].get("min_lr", 1e-6)
            )

            for epoch in range(1, epochs + 1):
                self.current_epoch += 1
                t0 = time.time()

                # Handle backbone warm-up freezing
                if epoch <= freeze_epochs:
                    self.model.freeze_backbone(freeze=True)
                    freeze_status = "Frozen"
                else:
                    self.model.freeze_backbone(freeze=False)
                    freeze_status = "Unfrozen"

                # Train epoch
                trn_loss, trn_acc, trn_f1 = self.train_epoch(train_loader, optimizer)

                # Validation epoch
                val_loss, val_acc, val_prec, val_rec, val_f1 = self.evaluate(val_loader)
                scheduler.step()

                elapsed = time.time() - t0
                print(
                    f"Epoch {epoch:02d}/{epochs:02d} (Global {self.current_epoch:02d}) [{res}px - {freeze_status}] "
                    f"Trn Loss: {trn_loss:.4f} Acc: {trn_acc*100:.1f}% | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% F1: {val_f1*100:.1f}% | "
                    f"{elapsed:.1f}s"
                )

                # Track best checkpoint
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    checkpoint_data = {
                        "epoch": self.current_epoch,
                        "stage": stage_name,
                        "resolution": res,
                        "model_state_dict": self.model.state_dict(),
                        "best_val_f1": self.best_val_f1,
                        "class_to_idx": self.class_to_idx,
                        "config": self.config,
                    }
                    torch.save(checkpoint_data, self.best_checkpoint_path)
                    print(f"  >>> [SAVED BEST MODEL] New Peak Val F1: {self.best_val_f1*100:.2f}% at {self.best_checkpoint_path.name}")

                self.history.append({
                    "stage": stage_name,
                    "resolution": res,
                    "stage_epoch": epoch,
                    "global_epoch": self.current_epoch,
                    "train_loss": trn_loss,
                    "train_acc": trn_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_f1": val_f1,
                })

        print("\n" + "=" * 70)
        print(f"TRAINING COMPLETE. Peak Val Macro-F1: {self.best_val_f1*100:.2f}%")
        print(f"Best weights preserved at: {self.best_checkpoint_path}")
        print("=" * 70)

        return {
            "best_val_f1": self.best_val_f1,
            "best_checkpoint": str(self.best_checkpoint_path),
            "history": self.history,
        }
