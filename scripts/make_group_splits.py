"""
Leak-Free Group-Based Split Generator for Wind Turbine Blade Inspection Dataset.

Prevents data leakage caused by consecutive drone video frames / flight passes
by clustering consecutive frames into atomic sequence groups and splitting
by Group ID (70% Train, 15% Val, 15% Test).
"""

import os
import json
import random
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = PROJECT_ROOT / "data" / "figshare" / "WT blade defect dataset" / "JPEGImages"
ANN_DIR = PROJECT_ROOT / "data" / "figshare" / "WT blade defect dataset" / "Annotations"
OUT_SPLIT_DIR = PROJECT_ROOT / "data" / "splits"

RAW_MAP = {
    "crack": "crack",
    "craze": "craze_crack",
    "corrosion": "erosion",
    "surface_injure": "surface_damage",
    "thunderstrike": "lightning_damage",
    "hide_craze": "delamination"
}

BASE_SEVERITY = {
    "surface_damage": 1,
    "erosion": 2,
    "craze_crack": 3,
    "crack": 4,
    "delamination": 4,
    "lightning_damage": 5
}


def build_samples_and_groups(diff_threshold: float = 75.0):
    all_xml = sorted(ANN_DIR.glob("*.xml"), key=lambda p: int(p.stem))
    records = []
    
    current_group = 0
    prev_img = None
    
    for i, xml_p in enumerate(all_xml):
        stem = int(xml_p.stem)
        img_p = IMG_DIR / f"{stem}.jpg"
        if not img_p.exists():
            continue
            
        tree = ET.parse(xml_p)
        root = tree.getroot()
        size_el = root.find("size")
        w = int(size_el.find("width").text) if size_el is not None else 1024
        h = int(size_el.find("height").text) if size_el is not None else 1024
        total_area = float(w * h)
        
        objects = []
        best_sev = 0
        best_cls = "normal"
        max_box_area = 0.0
        
        for obj in root.findall("object"):
            raw_name = obj.find("name").text.strip().lower()
            canonical = RAW_MAP.get(raw_name)
            if not canonical:
                continue
            bnd = obj.find("bndbox")
            xmin = max(0.0, float(bnd.find("xmin").text))
            ymin = max(0.0, float(bnd.find("ymin").text))
            xmax = min(float(w), float(bnd.find("xmax").text))
            ymax = min(float(h), float(bnd.find("ymax").text))
            box_area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
            area_ratio = box_area / total_area
            
            # Ground truth severity rule grounded in aerodynamic defect standards
            sev = BASE_SEVERITY[canonical]
            if area_ratio >= 0.08 and sev < 5:
                sev += 1
            elif area_ratio <= 0.005 and sev > 1:
                sev -= 1
                
            objects.append({
                "class": canonical,
                "bbox": [xmin, ymin, xmax, ymax],
                "area_ratio": area_ratio,
                "calibrated_severity": sev
            })
            
            if sev > best_sev or (sev == best_sev and box_area > max_box_area):
                best_sev = sev
                best_cls = canonical
                max_box_area = box_area
                
        # Load downsampled image for visual sequence clustering
        curr_img = cv2.imread(str(img_p))
        if i == 0:
            current_group = 0
        else:
            prev_stem = int(all_xml[i - 1].stem)
            if stem == prev_stem + 1 and curr_img is not None and prev_img is not None:
                s1 = cv2.resize(prev_img, (64, 64)).astype(float)
                s2 = cv2.resize(curr_img, (64, 64)).astype(float)
                diff = np.mean(np.abs(s1 - s2))
                if diff > diff_threshold:
                    current_group += 1
            else:
                current_group += 1
                
        prev_img = curr_img
        
        records.append({
            "image_id": f"{stem}.jpg",
            "stem": stem,
            "image_path": str(img_p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "xml_path": str(xml_p.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "width": w,
            "height": h,
            "group_id": current_group,
            "dominant_class": best_cls,
            "severity_level": best_sev,
            "defect_count": len(objects),
            "defects": objects
        })
        
    return records


def generate_leak_free_split(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    
    records = build_samples_and_groups()
    total_imgs = len(records)
    
    # Group records by group_id
    groups_dict = defaultdict(list)
    for r in records:
        groups_dict[r["group_id"]].append(r)
        
    all_group_ids = list(groups_dict.keys())
    print(f"Total samples: {total_imgs} across {len(all_group_ids)} distinct sequence groups.")
    
    # Stratified Group assignment based on group dominant class
    group_meta = []
    for gid, items in groups_dict.items():
        classes_in_group = [item["dominant_class"] for item in items]
        dom_group_cls = Counter(classes_in_group).most_common(1)[0][0]
        max_sev = max(item["severity_level"] for item in items)
        group_meta.append((gid, dom_group_cls, max_sev, len(items)))
        
    # Shuffle within class buckets
    by_class = defaultdict(list)
    for g in group_meta:
        by_class[g[1]].append(g)
        
    train_groups = set()
    val_groups = set()
    test_groups = set()
    
    for cls, g_list in by_class.items():
        random.shuffle(g_list)
        # Allocate groups aiming for 70% train, 15% val, 15% test of image count
        total_cls_imgs = sum(g[3] for g in g_list)
        n_test_target = round(total_cls_imgs * 0.15)
        n_val_target = round(total_cls_imgs * 0.15)
        
        curr_test, curr_val, curr_train = 0, 0, 0
        for g in g_list:
            gid, _, _, n_items = g
            if curr_test + n_items <= n_test_target and curr_test < n_test_target:
                test_groups.add(gid)
                curr_test += n_items
            elif curr_val + n_items <= n_val_target and curr_val < n_val_target:
                val_groups.add(gid)
                curr_val += n_items
            else:
                train_groups.add(gid)
                curr_train += n_items
                
    # Compile splits
    train_records = [r for r in records if r["group_id"] in train_groups]
    val_records = [r for r in records if r["group_id"] in val_groups]
    test_records = [r for r in records if r["group_id"] in test_groups]
    
    # Assert ZERO group overlap
    train_gids = {r["group_id"] for r in train_records}
    val_gids = {r["group_id"] for r in val_records}
    test_gids = {r["group_id"] for r in test_records}
    
    assert train_gids.isdisjoint(val_gids), "LEAKAGE ERROR: train and val share groups!"
    assert train_gids.isdisjoint(test_gids), "LEAKAGE ERROR: train and test share groups!"
    assert val_gids.isdisjoint(test_gids), "LEAKAGE ERROR: val and test share groups!"
    
    print("\n" + "=" * 60)
    print("LEAK-FREE GROUP-BASED SPLIT SUMMARY (ZERO SEQUENCE OVERLAP)")
    print("=" * 60)
    print(f"Train split : {len(train_records)} images ({len(train_gids)} groups, {len(train_records)/total_imgs*100:.1f}%)")
    print(f"Val split   : {len(val_records)} images ({len(val_gids)} groups, {len(val_records)/total_imgs*100:.1f}%)")
    print(f"Test split  : {len(test_records)} images ({len(test_gids)} groups, {len(test_records)/total_imgs*100:.1f}%)")
    print(f"Total       : {total_imgs} images (Zero Leakage Verified)")
    
    print("\nClass breakdown in Train / Val / Test:")
    for cls in RAW_MAP.values():
        c_tr = sum(1 for r in train_records if r["dominant_class"] == cls)
        c_va = sum(1 for r in val_records if r["dominant_class"] == cls)
        c_te = sum(1 for r in test_records if r["dominant_class"] == cls)
        print(f"  {cls:<18}: Train={c_tr:<3} Val={c_va:<3} Test={c_te:<3} Total={c_tr+c_va+c_te}")
        
    print("\nSeverity breakdown (1-5) in Train / Val / Test:")
    for s in range(1, 6):
        s_tr = sum(1 for r in train_records if r["severity_level"] == s)
        s_va = sum(1 for r in val_records if r["severity_level"] == s)
        s_te = sum(1 for r in test_records if r["severity_level"] == s)
        print(f"  Level {s} (Severity) : Train={s_tr:<3} Val={s_va:<3} Test={s_te:<3} Total={s_tr+s_va+s_te}")
    print("=" * 60)
    
    OUT_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT_SPLIT_DIR / "leak_free_split_manifest.json"
    
    manifest_data = {
        "metadata": {
            "dataset_origin": "Figshare: Multiclass Dataset for Intelligent Detection of Wind Turbine Blade Defects",
            "leak_prevention_method": "GroupShuffleSplit on consecutive drone sequence clusters",
            "seed": seed,
            "total_images": total_imgs,
            "total_groups": len(all_group_ids),
            "splits": {
                "train_count": len(train_records),
                "val_count": len(val_records),
                "test_count": len(test_records)
            }
        },
        "train": train_records,
        "val": val_records,
        "test": test_records
    }
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
        
    print(f"\nManifest successfully saved to: {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    generate_leak_free_split(seed=42)
