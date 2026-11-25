"""
Severity sanity check: does the learned severity head beat a trivial lookup?

Lookup baseline: predict the defect class with the classifier, then output the
base severity of that class (BASE_SEVERITY in scripts/make_group_splits.py),
ignoring defect size. If the severity head only memorised the class->level
table, both MAEs would be about equal.

Usage: python scripts/severity_baseline.py
"""
import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.blade_multitask_net import TurbineBladeMultiTaskNet
from datasets.multitask_dataset import TurbineBladeMultiTaskDataset, CLASSES
from datasets.augmentations import get_val_transforms
from scripts.make_group_splits import BASE_SEVERITY


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = TurbineBladeMultiTaskDataset(split="test", transform=get_val_transforms(image_size=224), image_size=224)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    model = TurbineBladeMultiTaskNet(num_classes=6, num_ordinal_thresholds=4, backbone_name="resnet50",
                                     pretrained=False, use_cbam=True, dropout=0.3).to(device)
    ckpt = torch.load(PROJECT_ROOT / "outputs/proposed_multitask_cbam/checkpoints/best.pth", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    p_cls, t_cls, p_sev, t_sev = [], [], [], []
    for x, tt, ts, _ in loader:
        out = model(x.to(device))
        p_cls += out["type_logits"].argmax(1).cpu().tolist(); t_cls += tt.tolist(); t_sev += ts.tolist()
        p_sev += (1 + (torch.sigmoid(out["ordinal_logits"]) > 0.5).long().sum(-1)).cpu().tolist()
    t_sev = np.array(t_sev).round().astype(int); p_sev = np.array(p_sev)
    lookup_pred = np.array([BASE_SEVERITY[CLASSES[c]] for c in p_cls])
    lookup_true = np.array([BASE_SEVERITY[CLASSES[c]] for c in t_cls])
    print(f"Test images: {len(t_sev)}")
    print(f"Severity head (ordinal)           MAE = {mean_absolute_error(t_sev, p_sev):.3f} | exact = {(p_sev==t_sev).mean()*100:.1f}%")
    print(f"Lookup on PREDICTED class         MAE = {mean_absolute_error(t_sev, lookup_pred):.3f} | exact = {(lookup_pred==t_sev).mean()*100:.1f}%")
    print(f"Lookup on TRUE class (upper bound)MAE = {mean_absolute_error(t_sev, lookup_true):.3f} | exact = {(lookup_true==t_sev).mean()*100:.1f}%")
    print(f"Predict-the-mode (always level {np.bincount(t_sev).argmax()}) MAE = {mean_absolute_error(t_sev, np.full_like(t_sev, np.bincount(t_sev).argmax())):.3f}")
    cm = confusion_matrix(t_sev, p_sev, labels=[1, 2, 3, 4, 5])
    print("Severity confusion (rows true L1..L5, cols pred L1..L5):\n", cm)
    off = [(i+1, j+1, cm[i, j]) for i in range(5) for j in range(5) if abs(i-j) >= 2 and cm[i, j]]
    print("Errors of 2+ levels (true, pred, count):", off)


if __name__ == "__main__":
    main()
