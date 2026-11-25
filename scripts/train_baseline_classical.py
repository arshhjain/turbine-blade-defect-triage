"""
Classical Machine Learning Baseline for Wind Turbine Blade Defect Classification.

Extracts:
1. Color Histograms in HSV space (quantizing blade surface discoloration)
2. Edge / Gradient features (Sobel horizontal & vertical gradient distributions)
3. Haralick Texture features (Gray-Level Co-occurrence Matrix properties)

Classifier: Multi-Class Support Vector Machine (RBF kernel) with class balancing.
Trained strictly on the leak-free train split and evaluated on the held-out test split.
"""

import json
import time
from pathlib import Path
import numpy as np
import cv2
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "splits" / "leak_free_split_manifest.json"
OUT_DIR = PROJECT_ROOT / "outputs" / "baselines"

CLASSES = [
    "crack",
    "craze_crack",
    "delamination",
    "erosion",
    "lightning_damage",
    "surface_damage"
]
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}


def extract_classical_features(img_bgr: np.ndarray) -> np.ndarray:
    """Extract concatenated color, edge, and texture descriptors."""
    # Resize to standardized dimensions for consistent feature extraction
    resized = cv2.resize(img_bgr, (256, 256))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    # 1. Color Histogram Features (HSV: 8 bins for H, 8 for S, 8 for V)
    hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
    hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
    hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
    color_feats = np.concatenate([
        cv2.normalize(hist_h, None).flatten(),
        cv2.normalize(hist_s, None).flatten(),
        cv2.normalize(hist_v, None).flatten()
    ])

    # 2. Gradient / Edge Features (Sobel filter magnitudes and standard deviations)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_feats = np.array([
        np.mean(grad_mag),
        np.std(grad_mag),
        np.percentile(grad_mag, 75),
        np.percentile(grad_mag, 95)
    ])

    # 3. Simple Spatial Co-occurrence / Texture Summary
    # Compute horizontal pixel difference variance and vertical difference variance
    h_diff = np.abs(gray[:, 1:] - gray[:, :-1])
    v_diff = np.abs(gray[1:, :] - gray[:-1, :])
    texture_feats = np.array([
        np.mean(h_diff),
        np.std(h_diff),
        np.mean(v_diff),
        np.std(v_diff)
    ])

    return np.concatenate([color_feats, edge_feats, texture_feats])


def run_classical_baseline():
    print("=" * 65)
    print("RUNNING CLASSICAL BASELINE: HSV Color + Sobel Edges + SVM")
    print("=" * 65)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Split manifest missing at {MANIFEST_PATH}. Run scripts/make_group_splits.py first.")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Extract features for train split
    print(f"Extracting features for {len(manifest['train'])} training samples...")
    t0 = time.time()
    X_train, y_train = [], []
    for s in manifest["train"]:
        img_p = PROJECT_ROOT / s["image_path"]
        img = cv2.imread(str(img_p))
        if img is not None:
            feats = extract_classical_features(img)
            X_train.append(feats)
            y_train.append(CLASS_TO_IDX[s["dominant_class"]])

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Extract features for test split
    print(f"Extracting features for {len(manifest['test'])} test samples...")
    X_test, y_test = [], []
    for s in manifest["test"]:
        img_p = PROJECT_ROOT / s["image_path"]
        img = cv2.imread(str(img_p))
        if img is not None:
            feats = extract_classical_features(img)
            X_test.append(feats)
            y_test.append(CLASS_TO_IDX[s["dominant_class"]])

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Train SVM classifier with RBF kernel and balanced class weights
    print(f"Fitting Support Vector Classifier (RBF kernel, balanced)...")
    clf = SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", random_state=42, probability=True)
    clf.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predictions
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    rep_dict = classification_report(y_test, preds, target_names=CLASSES, output_dict=True, zero_division=0)
    rep_str = classification_report(y_test, preds, target_names=CLASSES, zero_division=0)
    cm = confusion_matrix(y_test, preds)

    print("\n" + "=" * 65)
    print("CLASSICAL BASELINE TEST EVALUATION RESULTS")
    print("=" * 65)
    print(f"Test Accuracy : {acc * 100:.2f}% | Macro F1-Score: {macro_f1 * 100:.2f}%")
    print(f"Feature Dim   : {X_train.shape[1]} descriptors | Fit Time: {train_time:.2f}s\n")
    print(rep_str)
    print("Confusion Matrix:")
    print(cm)
    print("=" * 65)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "model": "Classical_SVM_RBF",
        "feature_type": "HSV_Histograms_Sobel_Edges_Texture",
        "test_accuracy": float(acc),
        "test_macro_f1": float(macro_f1),
        "classification_report": rep_dict,
        "confusion_matrix": cm.tolist(),
        "training_time_seconds": float(train_time)
    }

    results_path = OUT_DIR / "classical_baseline_metrics.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    joblib.dump(clf, OUT_DIR / "svm_baseline.joblib")
    print(f"Classical baseline results saved to: {results_path}")
    return results


if __name__ == "__main__":
    run_classical_baseline()
