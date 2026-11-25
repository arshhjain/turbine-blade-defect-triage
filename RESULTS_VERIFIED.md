# Verified Results (single source of truth)

Recomputed on 22 Nov 2026 directly from the saved checkpoints by:

```powershell
python scripts/verify_test_metrics.py      # classification, confusion matrices, misclassified images, McNemar
python scripts/severity_baseline.py        # severity head vs. lookup baselines
```

Logs: `outputs/logs/verify_test_metrics.log`, `outputs/logs/severity_baseline.log`.
If any other document disagrees with this file, **this file is correct**.

## Setup shared by all deep models
- Data: Figshare WT blade defect dataset, 1,065 images, 6 classes, one label per image (highest-severity defect in the image).
- Split: `data/splits/leak_free_split_manifest.json`, 747 / 159 / 159 images, grouped into 361 near-duplicate groups; no group appears in two splits.
- Identical code, seed 42, 224x224 input, batch 16, 12 epochs, AdamW (lr 3e-4, wd 1e-4), cosine LR to 1e-6, ImageNet-pretrained ResNet50, fully fine-tuned.
- Best epoch chosen on **validation** macro-F1; test set used once at the end.

## 1. Defect type classification (test set, N = 159)

| Model | What changes | Accuracy | Macro-F1 | Errors |
|---|---|:---:|:---:|:---:|
| Classical: HSV + Sobel features + RBF SVM | no deep learning | 35.85% | 33.22% | 102 |
| **A** ResNet50, cross-entropy | deep baseline | 93.71% | 93.14% | 10 |
| **B** A + CBAM attention | attention only | **94.34%** | **93.75%** | 9 |
| **C** B + focal loss + severity heads (proposed) | multi-task | 92.45% | 91.61% | 12 |

**McNemar test, C vs A:** C alone right on 2 images, A alone right on 4, exact p = 0.69.
**Interpretation:** deep features crush hand-crafted features (35.85% → ~93%). The differences between A, B and C are 1–3 images out of 159 and are **not statistically significant**. CBAM and multi-task learning did **not** measurably improve classification on this test set. A single seed and 159 test images cannot resolve differences this small.

## 2. Severity prediction (model C only, test set)

| Method | MAE (levels) | Exact level |
|---|:---:|:---:|
| Always predict the most common level (4) | 0.981 | — |
| Look up base severity of **predicted** class | 0.535 | 53.5% |
| Look up base severity of **true** class (best a lookup can do) | 0.415 | 58.5% |
| **Learned ordinal severity head** | **0.346** | **71.1%** |

**Interpretation:** the severity head beats even a perfect-class lookup, so it has learned the defect-size component of the label rule, not just class → level.
**Caveat:** severity labels are my own proxy rule (class base level, ±1 by box-area ratio), not dataset labels and not an industry standard.

Severity confusion (rows = true L1..L5, cols = predicted):
```
[[14  3  0  1  0]
 [ 1 11  6  1  0]
 [ 0  1 16  6  0]
 [ 2  0  1 48  7]
 [ 0  1  0 16 24]]
```
5 errors are 2+ levels off: L1→L4 (1), L2→L4 (1), **L4→L1 (2)**, **L5→L2 (1)**. The last three are dangerous under-estimates. The most common error is L5 → L4 (16 images): the model tends to under-call critical severity.

## 3. Failure analysis (model C, 12 misclassified test images)
- **Lightning damage is the weakest class** (recall 73%, 4 of 15 missed: 77, 95, 754, 834). Two of the misses were predicted as `craze_crack` (three for models A and B); burn marks with cracked edges look like crack networks. It is also the rarest class (15 test images).
- **`craze_crack` absorbs errors** (7 of 12 wrong predictions are "craze_crack"; precision 79%). Fine-texture classes overlap.
- **Delamination → surface_damage** (14, 764): delamination's original dataset label is `hide_craze`, which is visually subtle.
- Images 14, 77, 95, 754, 764 are wrong for **all three** deep models, so these are hard or ambiguous images, not model-specific failures. Use them in the demo as honest failure cases.

## Scope and closed-set limitation
All 1,065 dataset images contain at least one defect; there are **no healthy blades**. The softmax must therefore
choose one of the 6 defect classes for any input, including a pristine blade (observed on 23 Nov with an uploaded
photo of a new blade, predicted as `crack`). The system is scoped as **triage of already-reported damage**:
type -> repair team, severity -> priority. Future work: a stage-1 anomaly detector (e.g. PatchCore-style
nearest-neighbour distance on ResNet features of defect-free regions) to filter healthy blades before this model.

## Severity labels depend on camera distance
The proxy rule shifts severity by box area as a fraction of the image. Box area in pixels depends on how far the
drone was, so the rule measures **apparent** size. It changed the label (vs. the class base level) on 473 of 1,065
images (463 up, 10 down); the 8% threshold sits near the median largest-box ratio (7.6%).
Deployment fix: a capture SOP (fixed stand-off distance, camera and zoom) so pixel area is proportional to real area;
the model and thresholds would then be retrained/recalibrated on SOP-captured images. Alternative: convert pixels to
cm^2 using ground sample distance from drone altitude (not available in this dataset).
Better long-term measure: damage size relative to the blade's local width (chord) in the same image; both scale
equally with distance, so the ratio is distance-invariant. Requires a blade segmentation model (not implemented).

## Demo images (all from the TEST split, never seen in training)
| Image | True class | True severity | Model C prediction |
|---|---|:---:|---|
| `10.jpg` | lightning_damage | 5 | correct (lightning, L5) |
| `4.jpg` | crack | 5 | correct (crack 89.5%, L5) |
| `3.jpg` | erosion | 3 | correct (erosion 94.6%, L3) |
| `754.jpg` | lightning_damage | 5 | **wrong**: craze_crack at only 46% confidence, L4 (failure case) |

Note: images 104, 42 and 1000 used in earlier drafts are in the **training** split; do not call them held-out.

## Superseded
`outputs/baseline_b_vanilla_resnet50` (91.19%) was trained with an earlier version of the model code (BatchNorm bottleneck) and is not a controlled comparison. Replaced by run A.
