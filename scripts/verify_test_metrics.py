"""
Recompute held-out TEST metrics directly from saved checkpoints (no retraining).

Purpose: live, reproducible evidence for the panel that the reported numbers
come from the saved weights, plus a list of misclassified test images for
failure analysis, plus a McNemar test comparing the two deep models.

Usage:
    python scripts/verify_test_metrics.py
"""
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import sys
import json
from math import comb
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.blade_multitask_net import TurbineBladeMultiTaskNet
from datasets.multitask_dataset import TurbineBladeMultiTaskDataset, CLASSES
from datasets.augmentations import get_val_transforms

# Controlled ablation: identical code, split, seed (42), 224 px, 12 epochs, AdamW 3e-4, cosine LR.
# (The older outputs/baseline_b_vanilla_resnet50 checkpoint used a BatchNorm bottleneck from an
#  earlier code version, so it is not comparable and is superseded by run A.)
RUNS = {
    "ablation_A_vanilla_resnet50": dict(use_cbam=False, multitask=False),  # A: ResNet50, cross-entropy
    "ablation_B_resnet50_cbam": dict(use_cbam=True, multitask=False),      # B: A + CBAM
    "proposed_multitask_cbam": dict(use_cbam=True, multitask=True),        # C: B + focal loss + severity heads
}
COMPARE = ("proposed_multitask_cbam", "ablation_A_vanilla_resnet50")


@torch.no_grad()
def predict(run_name, cfg, loader, ids, device):
    model = TurbineBladeMultiTaskNet(num_classes=len(CLASSES), num_ordinal_thresholds=4,
                                     backbone_name="resnet50", pretrained=False,
                                     use_cbam=cfg["use_cbam"], dropout=0.3).to(device)
    ckpt = torch.load(PROJECT_ROOT / "outputs" / run_name / "checkpoints" / "best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    p_type, t_type, p_sev, t_sev = [], [], [], []
    for images, tt, ts, _ in loader:
        out = model(images.to(device))
        p_type += torch.argmax(out["type_logits"], 1).cpu().tolist()
        t_type += tt.tolist()
        t_sev += ts.tolist()
        if cfg["multitask"]:
            p_sev += (1 + (torch.sigmoid(out["ordinal_logits"]) > 0.5).long().sum(-1)).cpu().tolist()
    del model
    torch.cuda.empty_cache()
    return np.array(p_type), np.array(t_type), np.array(p_sev), np.array(t_sev)


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p-value. b = only model A correct, c = only model B correct."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TurbineBladeMultiTaskDataset(split="test", transform=get_val_transforms(image_size=224), image_size=224)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    manifest = json.load(open(PROJECT_ROOT / "data" / "splits" / "leak_free_split_manifest.json"))
    ids = [r["image_id"] for r in manifest["test"]]
    print(f"Device: {device} | Held-out test images: {len(ds)}\n")

    correct = {}
    for run, cfg in RUNS.items():
        pt, tt, ps, ts = predict(run, cfg, loader, ids, device)
        correct[run] = pt == tt
        print("=" * 70)
        print(f"{run}")
        print(f"Accuracy {accuracy_score(tt, pt)*100:.2f}% | Macro-F1 {f1_score(tt, pt, average='macro')*100:.2f}%"
              + (f" | Severity MAE {mean_absolute_error(ts, ps):.3f}" if len(ps) else ""))
        print(classification_report(tt, pt, labels=list(range(len(CLASSES))), target_names=CLASSES, digits=3, zero_division=0))
        print("Confusion matrix (rows = true, cols = predicted):", CLASSES)
        print(confusion_matrix(tt, pt, labels=list(range(len(CLASSES)))))
        wrong = [(ids[i], CLASSES[tt[i]], CLASSES[pt[i]]) for i in np.where(pt != tt)[0]] if len(ids) == len(tt) else []
        print(f"\nMisclassified test images ({len(wrong)}):")
        for img, t, p in wrong:
            print(f"  {img:<10} true={t:<17} pred={p}")
        print()

    a, b = correct[COMPARE[0]], correct[COMPARE[1]]
    only_prop, only_base = int((a & ~b).sum()), int((~a & b).sum())
    print("=" * 70)
    print(f"McNemar test (proposed vs vanilla): only-proposed-correct={only_prop}, "
          f"only-vanilla-correct={only_base}, exact p={mcnemar_exact(only_prop, only_base):.3f}")
    print("p >= 0.05 means the accuracy difference is NOT statistically significant on this test set.")


if __name__ == "__main__":
    main()
