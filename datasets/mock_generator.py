"""
Procedural Synthetic Wind Turbine Blade Drone Image Generator.

Generates realistic wind turbine blade inspection images with procedural defects
(cracking, leading-edge erosion, composite delamination, and pristine normal surfaces)
for rapid end-to-end testing, validation, and benchmarking of the MVP pipeline.
"""

import os
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np


def draw_blade_background(h: int, w: int) -> np.ndarray:
    """Generate realistic wind turbine blade aerial background (sky/clouds + curved blade surface)."""
    # Aerial sky/horizon gradient
    img = np.zeros((h, w, 3), dtype=np.uint8)
    sky_top = np.array([210, 180, 140])      # Light blue sky (BGR)
    sky_bottom = np.array([245, 220, 180])   # Horizon haze
    for y in range(h):
        ratio = y / h
        img[y, :] = (sky_top * (1 - ratio) + sky_bottom * ratio).astype(np.uint8)

    # Add subtle atmospheric cloud noise
    noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw curved aerodynamic fiberglass turbine blade (spanning diagonally across field of view)
    pts = np.array([
        [int(w * 0.15), 0],
        [int(w * 0.85), 0],
        [int(w * 0.95), h],
        [int(w * 0.35), h],
    ], np.int32)

    # Blade composite gel-coat base (light gray/off-white with aerodynamic curvature shading)
    blade_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(blade_mask, [pts], 255)

    blade_texture = np.ones((h, w, 3), dtype=np.uint8) * 230
    # Add subtle longitudinal fiber texture lines
    for x in range(0, w, 12):
        cv2.line(blade_texture, (x, 0), (x + int(w * 0.1), h), (220, 220, 220), 1)

    # Apply aerodynamic curvature shadow
    for y in range(h):
        shade = int(25 * np.sin(np.pi * y / h))
        blade_texture[y, :] = np.clip(blade_texture[y, :].astype(np.int16) - shade, 0, 255).astype(np.uint8)

    # Blend blade onto sky background
    mask_3ch = cv2.merge([blade_mask, blade_mask, blade_mask])
    img = np.where(mask_3ch == 255, blade_texture, img)

    # Draw leading edge boundary line
    cv2.polylines(img, [pts], False, (170, 170, 170), 2)
    return img


def inject_cracking_defect(img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Procedurally render a jagged hairline structural crack with micro-branches."""
    h, w, _ = img.shape
    start_x = np.random.randint(int(w * 0.35), int(w * 0.65))
    start_y = np.random.randint(int(h * 0.20), int(h * 0.40))

    curr_x, curr_y = start_x, start_y
    pts = [(curr_x, curr_y)]
    steps = np.random.randint(25, 45)

    for _ in range(steps):
        curr_x += np.random.randint(-4, 5)
        curr_y += np.random.randint(4, 10)
        pts.append((curr_x, curr_y))

    # Render main fracture line
    pts_arr = np.array(pts, np.int32)
    cv2.polylines(img, [pts_arr], False, (25, 25, 25), 2)
    # Add micro-branches
    for i in range(5, len(pts) - 5, 6):
        bx, by = pts[i]
        b_end = (bx + np.random.randint(-15, 16), by + np.random.randint(5, 15))
        cv2.line(img, (bx, by), b_end, (40, 40, 40), 1)

    # Bounding box calculation
    min_x = max(0, min(p[0] for p in pts) - 10)
    max_x = min(w, max(p[0] for p in pts) + 10)
    min_y = max(0, min(p[1] for p in pts) - 10)
    max_y = min(h, max(p[1] for p in pts) + 10)

    bbox_norm = (min_x / w, min_y / h, max_x / w, max_y / h)
    return img, bbox_norm


def inject_erosion_defect(img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Procedurally render leading-edge erosion (pitting, gel coat peeling, surface wear)."""
    h, w, _ = img.shape
    center_x = np.random.randint(int(w * 0.45), int(w * 0.70))
    center_y = np.random.randint(int(h * 0.30), int(h * 0.60))
    radius_x = np.random.randint(25, 55)
    radius_y = np.random.randint(40, 90)

    # Draw localized rough erosion patch (darker fiberglass substrate exposure)
    patch_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(patch_mask, (center_x, center_y), (radius_x, radius_y), 25, 0, 360, 255, -1)

    # Micro-pitting specks
    specks = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    specks = cv2.threshold(specks, 210, 255, cv2.THRESH_BINARY)[1]
    erosion_specks = cv2.bitwise_and(patch_mask, specks)

    # Apply brownish/dark gray wear texture
    wear_color = np.array([90, 110, 130], dtype=np.uint8)
    img[erosion_specks > 0] = wear_color

    # Add abrasion boundary
    cv2.ellipse(img, (center_x, center_y), (radius_x, radius_y), 25, 0, 360, (140, 150, 160), 2)

    min_x = max(0, center_x - radius_x - 5)
    max_x = min(w, center_x + radius_x + 5)
    min_y = max(0, center_y - radius_y - 5)
    max_y = min(h, center_y + radius_y + 5)

    bbox_norm = (min_x / w, min_y / h, max_x / w, max_y / h)
    return img, bbox_norm


def inject_delamination_defect(img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Procedurally render composite delamination (blistering, layer buckling, disbonding)."""
    h, w, _ = img.shape
    center_x = np.random.randint(int(w * 0.40), int(w * 0.65))
    center_y = np.random.randint(int(h * 0.35), int(h * 0.65))
    axes = (np.random.randint(35, 70), np.random.randint(25, 50))
    angle = np.random.randint(-30, 30)

    # Delamination blister outline & surface discoloration
    overlay = img.copy()
    cv2.ellipse(overlay, (center_x, center_y), axes, angle, 0, 360, (160, 175, 195), -1)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    # Distinct separation contour ring
    cv2.ellipse(img, (center_x, center_y), axes, angle, 0, 360, (70, 70, 70), 2)
    # Secondary internal buckle crease
    inner_axes = (int(axes[0] * 0.6), int(axes[1] * 0.5))
    cv2.ellipse(img, (center_x + 3, center_y - 2), inner_axes, angle, 0, 360, (100, 100, 100), 1)

    min_x = max(0, center_x - axes[0] - 8)
    max_x = min(w, center_x + axes[0] + 8)
    min_y = max(0, center_y - axes[1] - 8)
    max_y = min(h, center_y + axes[1] + 8)

    bbox_norm = (min_x / w, min_y / h, max_x / w, max_y / h)
    return img, bbox_norm


def generate_synthetic_turbine_dataset(
    output_dir: str,
    samples_per_class: int = 30,
    img_size: int = 512,
    create_roboflow_yolo: bool = True,
) -> None:
    """
    Generate a full synthetic benchmark dataset with realistic blade defect imagery.
    Produces both directory-per-class and Roboflow YOLO-style annotation structures.
    """
    classes = ["normal", "cracking", "erosion", "delamination"]
    out_path = Path(output_dir)

    for cls in classes:
        cls_dir = out_path / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

    # Optional Roboflow YOLO format
    yolo_img_dir = out_path / "roboflow" / "images"
    yolo_lbl_dir = out_path / "roboflow" / "labels"
    if create_roboflow_yolo:
        yolo_img_dir.mkdir(parents=True, exist_ok=True)
        yolo_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating synthetic turbine blade dataset ({samples_per_class} per class) at: {output_dir}")

    total_count = 0
    for cls_idx, cls in enumerate(classes):
        for i in range(samples_per_class):
            img = draw_blade_background(img_size, img_size)
            bbox = None

            if cls == "cracking":
                img, bbox = inject_cracking_defect(img)
            elif cls == "erosion":
                img, bbox = inject_erosion_defect(img)
            elif cls == "delamination":
                img, bbox = inject_delamination_defect(img)
            elif cls == "normal":
                bbox = None  # Pristine blade

            # Save in class directory
            filename = f"blade_{cls}_{i:04d}.jpg"
            img_save_path = out_path / cls / filename
            cv2.imwrite(str(img_save_path), img)

            # Save in Roboflow YOLO format
            if create_roboflow_yolo:
                yolo_img_path = yolo_img_dir / filename
                cv2.imwrite(str(yolo_img_path), img)

                yolo_txt_path = yolo_lbl_dir / f"blade_{cls}_{i:04d}.txt"
                if bbox is not None:
                    bx1, by1, bx2, by2 = bbox
                    xc = (bx1 + bx2) / 2.0
                    yc = (by1 + by2) / 2.0
                    bw = bx2 - bx1
                    bh = by2 - by1
                    with open(yolo_txt_path, "w", encoding="utf-8") as f:
                        f.write(f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                else:
                    # Normal blade has empty annotation file
                    with open(yolo_txt_path, "w", encoding="utf-8") as f:
                        pass

            total_count += 1

    # Write Roboflow data.yaml
    if create_roboflow_yolo:
        yaml_content = f"""# Roboflow Wind Turbine Blade Defect Dataset Config
path: {out_path / 'roboflow'}
train: images
val: images
names:
  0: normal
  1: cracking
  2: erosion
  3: delamination
"""
        with open(out_path / "roboflow" / "data.yaml", "w", encoding="utf-8") as f:
            f.write(yaml_content)

    print(f"Successfully generated {total_count} synthetic turbine inspection samples.")
