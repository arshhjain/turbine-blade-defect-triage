"""
Actionable Diagnostic Assessment Engine for Wind Turbine Blade Defect Classification.

Produces immediate, production-ready inspection reports and severity assessments
suitable for two-day project submission and field deployment.
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from datasets.augmentations import get_val_transforms
from datasets.unified_loader import CANONICAL_CLASS_MAP
from .gradcam import GradCAM


def find_xml_annotation(image_path: Path) -> Optional[Path]:
    """Find corresponding Pascal VOC XML annotation file if available."""
    # Check parent.parent / Annotations / stem.xml (Figshare structure)
    candidate1 = image_path.parent.parent / "Annotations" / f"{image_path.stem}.xml"
    if candidate1.exists():
        return candidate1
    # Check same directory
    candidate2 = image_path.parent / f"{image_path.stem}.xml"
    if candidate2.exists():
        return candidate2
    return None


def parse_xml_annotations(xml_path: Optional[Path]) -> List[Dict[str, Any]]:
    """Parse defect instances and bounding boxes from Pascal VOC XML."""
    objects = []
    if not xml_path or not xml_path.exists():
        return objects

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name_el = obj.find("name")
            if name_el is None or not name_el.text:
                continue
            raw_name = name_el.text.strip().lower()
            canonical_name = CANONICAL_CLASS_MAP.get(raw_name, raw_name)

            bnd = obj.find("bndbox")
            box = None
            if bnd is not None:
                box = [
                    int(float(bnd.find("xmin").text)),
                    int(float(bnd.find("ymin").text)),
                    int(float(bnd.find("xmax").text)),
                    int(float(bnd.find("ymax").text)),
                ]
            objects.append({
                "raw_name": raw_name,
                "canonical_name": canonical_name,
                "bbox": box,
            })
    except Exception as e:
        print(f"[Warning] Failed to parse XML annotation {xml_path}: {e}")

    return objects


# Severity Matrix and Operational Action Protocols
SEVERITY_CATALOG = {
    "cracking": {
        "level": 4,
        "urgency": "HIGH",
        "title": "Structural Substrate Fracture",
        "risk_mechanism": "Stress concentration along trailing/leading edge; high probability of crack propagation into primary spar under cyclical aerodynamic gusts.",
        "recommended_action": "Schedule rope-access non-destructive ultrasonic testing (NDT) within 48 to 72 hours. Curtail turbine operational pitch angle if wind gusts exceed 12 m/s.",
    },
    "delamination": {
        "level": 4,
        "urgency": "CRITICAL",
        "title": "Composite Laminate Disbonding / Blistering",
        "risk_mechanism": "Separation of outer skin from structural core or composite shear web; compromises blade aerodynamic profile and risks shell buckling.",
        "recommended_action": "Immediate thermal imaging / tap-test inspection within 24 to 48 hours. Prepare vacuum-assisted resin infusion or scarf laminate patch repair.",
    },
    "erosion": {
        "level": 2,
        "urgency": "MODERATE",
        "title": "Leading-Edge Pitting & Aerodynamic Degradation",
        "risk_mechanism": "Progressive abrasive pitting of protective gel coat from rain/droplets, increasing drag and reducing annual energy production (AEP) by 3-5%.",
        "recommended_action": "Log for scheduled maintenance window. Apply leading-edge protection (LEP) polyurethane tape or specialized erosion-resistant coating.",
    },
    "normal": {
        "level": 0,
        "urgency": "NOMINAL",
        "title": "Healthy Aerodynamic Blade Surface",
        "risk_mechanism": "No surface discontinuities, fractures, or coating breakdown detected.",
        "recommended_action": "Retain within standard quarterly UAV aerial surveillance schedule.",
    },
}


class BladeDiagnosticEngine:
    """
    Automated drone inspection diagnostic engine.
    """
    def __init__(
        self,
        model: torch.nn.Module,
        class_to_idx: Dict[str, int],
        device: torch.device,
        image_size: int = 512,
        output_dir: str = "outputs/diagnostics",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.device = device
        self.image_size = image_size
        self.transform = get_val_transforms(image_size=image_size)

        raw_output_dir = Path(output_dir)
        if not raw_output_dir.is_absolute():
            project_root = Path(__file__).resolve().parent.parent
            self.output_dir = project_root / raw_output_dir
        else:
            self.output_dir = raw_output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize GradCAM
        self.gradcam = GradCAM(self.model)

    def create_comparison_image(
        self,
        original_rgb: np.ndarray,
        blended_cam_rgb: np.ndarray,
        diagnostic: Dict[str, Any],
        xml_objects: List[Dict[str, Any]],
        output_filename: str,
        pane_size: int = 768,
    ) -> str:
        """
        Create a side-by-side composite comparison image:
        [Input Drone Image with Ground Truth] | [Model Grad-CAM Attention Heatmap]
        """
        orig_h, orig_w = original_rgb.shape[:2]
        annotated_rgb = original_rgb.copy()

        # Draw ground truth bounding boxes if available
        for obj in xml_objects:
            box = obj.get("bbox")
            if box:
                xmin, ymin, xmax, ymax = box
                xmin = max(0, min(xmin, orig_w - 1))
                xmax = max(0, min(xmax, orig_w - 1))
                ymin = max(0, min(ymin, orig_h - 1))
                ymax = max(0, min(ymax, orig_h - 1))

                # Draw bounding box
                cv2.rectangle(annotated_rgb, (xmin, ymin), (xmax, ymax), (46, 204, 113), 3)

                # Draw label badge
                lbl = f"GT: {obj['canonical_name'].upper()} ({obj['raw_name']})"
                (tw, th), baseline = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                badge_top = max(0, ymin - th - 8)
                badge_bottom = ymin
                cv2.rectangle(annotated_rgb, (xmin, badge_top), (xmin + tw + 10, badge_bottom), (20, 24, 32), -1)
                cv2.rectangle(annotated_rgb, (xmin, badge_top), (xmin + tw + 10, badge_bottom), (46, 204, 113), 1)
                cv2.putText(annotated_rgb, lbl, (xmin + 5, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (46, 204, 113), 2, cv2.LINE_AA)

        # Resize panes
        left_pane = cv2.resize(annotated_rgb, (pane_size, pane_size), interpolation=cv2.INTER_AREA)
        right_pane = cv2.resize(blended_cam_rgb, (pane_size, pane_size), interpolation=cv2.INTER_AREA)

        # Top Headers (Height 60px)
        left_hdr = np.full((60, pane_size, 3), (24, 32, 45), dtype=np.uint8)
        left_title = f"INPUT UAV: {diagnostic['image_filename']}"
        cv2.putText(left_hdr, left_title, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        gt_classes = [o['canonical_name'].upper() for o in xml_objects]
        gt_str = f"GT: {', '.join(gt_classes)}" if gt_classes else "GT: UNANNOTATED"
        (gt_w, _), _ = cv2.getTextSize(gt_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(left_hdr, gt_str, (pane_size - gt_w - 18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (46, 204, 113), 2, cv2.LINE_AA)

        right_hdr = np.full((60, pane_size, 3), (24, 32, 45), dtype=np.uint8)
        right_title = f"AI OUTPUT: {diagnostic['primary_defect'].upper()} ({diagnostic['confidence_percentage']})"
        cv2.putText(right_hdr, right_title, (18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 225, 255), 2, cv2.LINE_AA)
        sev_str = f"SEVERITY: L{diagnostic['severity_level']} ({diagnostic['urgency']})"
        (sev_w, _), _ = cv2.getTextSize(sev_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(right_hdr, sev_str, (pane_size - sev_w - 18, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 195, 70), 2, cv2.LINE_AA)

        # Bottom Footer Banner (Height 80px, Width pane_size * 2)
        footer = np.full((80, pane_size * 2, 3), (16, 20, 28), dtype=np.uint8)
        if xml_objects:
            gt_set = {o['canonical_name'] for o in xml_objects}
            is_match = diagnostic['primary_defect'] in gt_set
            verdict_text = "VERDICT: MATCH [DEFECT DETECTED & LOCALIZED]" if is_match else "VERDICT: MISMATCH [FIELD REVIEW REQUIRED]"
            verdict_col = (46, 204, 113) if is_match else (255, 90, 70)
        else:
            verdict_text = "VERDICT: UNANNOTATED FRAME [INFERENCE COMPLETE]"
            verdict_col = (100, 200, 255)

        cv2.putText(footer, verdict_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, verdict_col, 2, cv2.LINE_AA)
        desc = f"Classified Anomaly: {diagnostic['anomaly_title']}"
        cv2.putText(footer, desc, (680, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)

        proto = f"Action Protocol: {diagnostic['actionable_protocol']}"
        if len(proto) > 135:
            proto = proto[:132] + "..."
        cv2.putText(footer, proto, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 185, 200), 1, cv2.LINE_AA)

        # Assemble composite
        left_col = np.vstack([left_hdr, left_pane])
        right_col = np.vstack([right_hdr, right_pane])
        body = np.hstack([left_col, right_col])
        composite = np.vstack([body, footer])

        # Save to disk
        out_path = self.output_dir / output_filename
        cv2.imwrite(str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        return str(out_path)

    def diagnose_image(
        self,
        image_path: str,
        generate_cam: bool = True,
        generate_comparison: bool = True,
        xml_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Perform complete diagnostic evaluation on a single drone image.

        Args:
            image_path: Path to RGB drone image file.
            generate_cam: If True, produces Grad-CAM overlay heatmap.
            generate_comparison: If True, produces side-by-side [Input | Output] comparison image.
            xml_path: Optional path to Pascal VOC XML annotation file.

        Returns:
            diagnostic_record: Structured diagnostic assessment dictionary.
        """
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Inspection image not found at: {image_path}")

        # Locate XML annotation
        resolved_xml = Path(xml_path) if xml_path else find_xml_annotation(img_path)
        xml_objects = parse_xml_annotations(resolved_xml)

        # Load RGB image
        original_bgr = cv2.imread(str(img_path))
        if original_bgr is None:
            raise ValueError(f"OpenCV could not decode image: {image_path}")
        original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

        # Apply transforms
        transformed = self.transform(image=original_rgb)
        tensor_img = transformed["image"].unsqueeze(0).to(self.device)

        # Run model inference
        with torch.enable_grad():
            tensor_img.requires_grad = True
            heatmap, pred_idx, confidence = self.gradcam.generate_heatmap(tensor_img)

        # Get all class probabilities
        with torch.no_grad():
            logits, _ = self.model(tensor_img)
            probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

        pred_class = self.idx_to_class.get(pred_idx, "unknown")
        class_probs = {self.idx_to_class[i]: float(probs[i]) for i in range(len(self.idx_to_class))}

        # Lookup severity protocol
        meta = SEVERITY_CATALOG.get(pred_class, {
            "level": 3,
            "urgency": "EVALUATE",
            "title": f"Unspecified Anomaly ({pred_class})",
            "risk_mechanism": "Surface anomaly detected requiring manual review.",
            "recommended_action": "Review UAV high-resolution telemetry.",
        })

        # Save Grad-CAM visual overlay
        cam_output_path = None
        blended = None
        if generate_cam or generate_comparison:
            blended = self.gradcam.overlay_heatmap(original_rgb, heatmap)
            # Add visual diagnostic banner
            banner_text = f"Defect: {pred_class.upper()} ({confidence*100:.1f}%) | Severity: Level {meta['level']}"
            cv2.putText(
                blended, banner_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA
            )
            cam_filename = f"diagnosis_cam_{img_path.stem}.jpg"
            cam_output_path = str(self.output_dir / cam_filename)
            cv2.imwrite(cam_output_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        # Ground truth matching
        gt_classes = list({o["canonical_name"] for o in xml_objects})
        is_match = (pred_class in gt_classes) if xml_objects else None

        # Format preliminary diagnostic assessment
        diagnostic = {
            "image_filename": img_path.name,
            "image_path": str(img_path),
            "primary_defect": pred_class,
            "confidence": round(confidence, 4),
            "confidence_percentage": f"{confidence * 100:.2f}%",
            "severity_level": meta["level"],
            "urgency": meta["urgency"],
            "anomaly_title": meta["title"],
            "risk_mechanism": meta["risk_mechanism"],
            "actionable_protocol": meta["recommended_action"],
            "all_class_probabilities": {k: round(v, 4) for k, v in class_probs.items()},
            "cam_overlay_saved_at": cam_output_path,
            "comparison_saved_at": None,
            "ground_truth_classes": gt_classes,
            "ground_truth_raw": [o["raw_name"] for o in xml_objects],
            "verdict_match": is_match,
        }

        # Generate side-by-side comparison image
        if generate_comparison and blended is not None:
            comp_filename = f"inspection_compare_{img_path.stem}.jpg"
            comp_path = self.create_comparison_image(
                original_rgb=original_rgb,
                blended_cam_rgb=blended,
                diagnostic=diagnostic,
                xml_objects=xml_objects,
                output_filename=comp_filename,
            )
            diagnostic["comparison_saved_at"] = comp_path

        return diagnostic

    def diagnose_batch(
        self,
        image_paths: List[str],
        save_json: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Diagnose a batch of drone inspection images and generate consolidated summary.
        """
        results = []
        for path in image_paths:
            try:
                diag = self.diagnose_image(path)
                results.append(diag)
            except Exception as e:
                results.append({
                    "image_filename": Path(path).name,
                    "error": str(e),
                })

        if save_json:
            json_path = self.output_dir / "drone_inspection_diagnosis_report.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Actionable inspection report written to: {json_path}")

        return results
