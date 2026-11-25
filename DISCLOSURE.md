# Academic Transparency & Integrity Disclosure
**Course:** Deep Learning (Semester V) — Project-Based Evaluation  
**Project Title:** Wind Turbine Blade Defect & Damage Severity Detection  
**Candidate:** Arsh Jain  
**Evaluation Date:** 23 November 2026  

---

## 1. Dataset Origin & Ground Truth Formulation
* **Primary Public Dataset:**
  * *Title:* Multiclass Dataset for Intelligent Detection of Wind Turbine Blade Defects Using Drone Imagery
  * *Source:* Figshare Open Repository (`data/figshare/WT blade defect dataset/`)
  * *Scale:* 1,065 high-resolution (1024×1024) RGB aerial inspection drone frames with Pascal VOC XML bounding box annotations.
  * *Defect Classes:* 6 physical anomaly types:
    1. `crack`: Structural through-thickness substrate fractures along leading/trailing edges.
    2. `craze_crack`: Dense micro-fissure networks from fatigue and UV resin breakdown.
    3. `delamination`: Composite laminate skin-core debonding, blistering, and internal voids.
    4. `erosion`: Leading-edge particulate pitting and gelcoat wear from rain and airborne grit.
    5. `lightning_damage`: High-energy strike burns, receptor detachment, and localized composite rupture.
    6. `surface_damage`: Superficial scratches, grease stains, and cosmetic paint spalling.
* **Ground Truth Severity Labeling Scheme:**
  * **Not from the dataset and not from any standard.** The dataset has no severity labels; this is a proxy rule I defined myself (base level per class, +/-1 by bounding-box area ratio), loosely inspired by blade-maintenance practice. Because the rule is a function of class + box size, the severity head partly learns to reproduce my rule; Severity MAE measures agreement with this rule, not real-world damage severity.
* **Class renaming:** the dataset's original labels were renamed: `crack`->crack, `craze`->craze_crack, `corrosion`->erosion, `surface_injure`->surface_damage, `thunderstrike`->lightning_damage, `hide_craze`->delamination. The `hide_craze`->delamination and `corrosion`->erosion mappings are my interpretation, not the dataset authors' terminology.
* **Image-level label:** images can contain several defects; each image is labelled with its highest-severity defect (ties broken by larger box).
  * Integer scale from Level 1 to Level 5:
    * Level 1 (Low): Superficial scuffs and cosmetic blemishes with zero structural risk.
    * Level 2 (Moderate): Leading-edge erosion and minor coating abrasions.
    * Level 3 (Elevated): Micro-fissures, extensive craze cracking requiring quarterly tracking.
    * Level 4 (High): Structural macro-cracks and composite laminate delaminations requiring NDT within 48h.
    * Level 5 (Critical): Catastrophic lightning arc strikes and deep structural splits requiring immediate turbine shutdown.
  * *Area-Calibrated Rule:* Escalated by $+1$ if the defect area ratio exceeds 8% of blade chord ($r \ge 0.08$), or decremented by $-1$ if below 0.5% ($r \le 0.005$, floored at 1).

---

## 2. Leak-Free Group Splitting Strategy
* **Sequence Clustering:**
  * Consecutively numbered frames are often near-duplicates of the same blade (assumed to come from continuous UAV passes). Random shuffling causes severe data leakage where consecutive frames from the same blade appear in both train and test splits.
  * *Remediation:* Frames were clustered into **361 atomic inspection sequence groups** using consecutive index continuity and structural visual distance.
  * *Manifest:* `data/splits/leak_free_split_manifest.json` partitions the 361 groups into:
    * **Train Split:** 747 images (258 groups, 70.1%)
    * **Validation Split:** 159 images (67 groups, 14.9%)
    * **Test Split:** 159 images (36 groups, 14.9%)
  * *Verification:* 100% group disjointness verified. Zero sequence, turbine, or background leakage across splits.

---

## 3. Pretrained Weights & Transfer Learning Strategy
* **Backbone Architecture:** ResNet50 (Residual Network with 50 layers, Bottleneck building blocks).
* **Pretrained Source:** ImageNet-1K (Torchvision `ResNet50_Weights.DEFAULT`, weights version V2).
* **Fine-Tuning Protocol:**
  * The whole network (all ResNet50 stages, CBAM and heads) is fine-tuned end-to-end with AdamW (lr 3e-4, weight decay 1e-4); nothing is frozen.
  * Learning rate follows cosine annealing (no warmup) down to 1e-6 over 12 epochs.
  * Batch Normalization running statistics ($\mu, \sigma^2$) are updated with momentum during training and frozen during evaluation to guarantee reproducible test-set inference.

---

## 4. External Code & Software Libraries
* **Deep Learning Framework:** PyTorch 2.6.0 with CUDA 11.8 on an NVIDIA GeForce RTX 3050 Laptop GPU.
* **Computer Vision & Augmentation:**
  * Albumentations 2.0.8: Heavy data augmentations (affine transforms, rotations, lighting shifts, motion blurs, atmospheric fog) applied strictly to the training split.
  * OpenCV (cv2) 4.11: Image decoding, color space transformations, and bounding box rendering.
* **Statistical Evaluation & Baselines:**
  * scikit-learn 1.7.2: Classification report, confusion matrices, and the classical SVM baseline (HSV Color + Sobel Edges + RBF SVM).
  * Matplotlib & Seaborn: Loss curves, confusion matrix heatmaps, and metric visualizations.

---

## 5. Disclosure of AI Assistance
* **AI Tools Used:** Antigravity (Google agentic coding assistant) and Claude Code (Anthropic), used on 22 Nov 2026 to audit the project, correct overstated claims in these documents, add `scripts/verify_test_metrics.py`, and run the controlled ablation (runs A and B).
* **Role of AI:**
  1. Automated code scaffolding and refactoring of the PyTorch training pipeline.
  2. Construction of the group-based sequence clustering script (`scripts/make_group_splits.py`) to eliminate data leakage.
  3. Formulation of the mathematical multi-task loss module (`training/multitask_losses.py`).
  4. Generation of technical viva defense questions and structured presentation scripts.
* **Candidate's Individual Responsibility & Ownership:**
  * The candidate (Arsh Jain) personally validated dataset splits, inspected training dynamics, reviewed loss gradients, executed baseline comparisons, and holds complete responsibility for explaining the mathematical derivations, architectural choices, and failure modes during the technical viva.
