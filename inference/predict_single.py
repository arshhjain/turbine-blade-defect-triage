"""
Standalone Offline CLI Inference Script for Wind Turbine Blade Defect & Severity Prediction.

Usage:
    python inference/predict_single.py --image "data/figshare/WT blade defect dataset/JPEGImages/104.jpg"
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.blade_multitask_net import TurbineBladeMultiTaskNet
from datasets.multitask_dataset import CLASSES, IDX_TO_CLASS
from datasets.augmentations import get_val_transforms

SEVERITY_PROTOCOLS = {
    1: {
        "urgency": "LOW / NOMINAL",
        "description": "Cosmetic gelcoat scuff or minor superficial coating wear.",
        "protocol": "Log anomaly in asset CMMS. Clean blade surface during next routine scheduled service."
    },
    2: {
        "urgency": "MODERATE",
        "description": "Leading-edge particulate pitting and aerodynamic gelcoat degradation.",
        "protocol": "Schedule leading-edge protection (LEP) polyurethane tape or elastomer repair at next quarterly service window."
    },
    3: {
        "urgency": "ELEVATED",
        "description": "Micro-fracture craze network from cyclical fatigue and UV exposure.",
        "protocol": "Perform close-range high-resolution drone inspection within 2 weeks. Sand back gelcoat and seal with UV-stabilized resin."
    },
    4: {
        "urgency": "HIGH",
        "description": "Structural through-crack or composite laminate delamination void.",
        "protocol": "Schedule rope-access ultrasonic non-destructive testing (NDT) within 48 to 72 hours. Curtail turbine pitch angle if gusts exceed 12 m/s."
    },
    5: {
        "urgency": "CRITICAL",
        "description": "High-energy lightning arc strike or explosive structural composite rupture.",
        "protocol": "GROUND TURBINE IMMEDIATELY. Perform internal borescope inspection of structural shear webs and grounding continuity test before restart."
    }
}


def load_model(checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    use_cbam = config.get("use_cbam", True)
    
    model = TurbineBladeMultiTaskNet(
        num_classes=len(CLASSES),
        num_ordinal_thresholds=4,
        backbone_name="resnet50",
        pretrained=False,
        use_cbam=use_cbam,
        dropout=0.0
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def draw_annotation_overlay(
    image_rgb: np.ndarray,
    pred_class: str,
    confidence: float,
    sev_level: int,
    sev_scalar: float,
    output_path: Path
):
    h, w = image_rgb.shape[:2]
    # Resize canvas to standardized 800x800 for consistent presentation
    canvas = cv2.resize(image_rgb, (800, 800))
    canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
    
    # Severity color: Level 1 (green), Level 2-3 (yellow/orange), Level 4-5 (red)
    color_map = {
        1: (80, 200, 50),
        2: (50, 200, 240),
        3: (0, 165, 255),
        4: (0, 80, 255),
        5: (30, 30, 230)
    }
    sev_color = color_map.get(sev_level, (0, 165, 255))
    
    # Header Banner
    header = np.zeros((90, 800, 3), dtype=np.uint8)
    header[:] = (25, 25, 30)
    
    cv2.putText(header, "WIND TURBINE BLADE DEFECT DIAGNOSIS", (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (240, 240, 240), 2, cv2.LINE_AA)
    
    subtext = f"Defect: {pred_class.upper()} ({confidence*100:.1f}%) | Severity: LEVEL {sev_level} (Score: {sev_scalar:.2f})"
    cv2.putText(header, subtext, (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, sev_color, 2, cv2.LINE_AA)
    
    # Footer Banner
    footer = np.zeros((100, 800, 3), dtype=np.uint8)
    footer[:] = (20, 20, 25)
    
    proto = SEVERITY_PROTOCOLS.get(sev_level, SEVERITY_PROTOCOLS[1])
    urgency_text = f"Action Urgency: {proto['urgency']}"
    cv2.putText(footer, urgency_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, sev_color, 2, cv2.LINE_AA)
    
    # Truncate protocol text if needed to fit
    action_text = f"Protocol: {proto['protocol']}"
    if len(action_text) > 75:
        line1 = action_text[:75]
        line2 = "          " + action_text[75:]
        cv2.putText(footer, line1, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(footer, line2, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(footer, action_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
        
    combined = np.vstack([header, canvas_bgr, footer])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)


def run_inference(image_path: str, checkpoint_path: Optional[str] = None):
    img_path = Path(image_path)
    if not img_path.is_absolute():
        img_path = PROJECT_ROOT / img_path
        
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: {img_path}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Default to proposed model checkpoint if available, else baseline
    if checkpoint_path is None:
        cand1 = PROJECT_ROOT / "outputs" / "proposed_multitask_cbam" / "checkpoints" / "best.pth"
        cand2 = PROJECT_ROOT / "outputs" / "baseline_b_vanilla_resnet50" / "checkpoints" / "best.pth"
        cand3 = PROJECT_ROOT / "outputs" / "checkpoints" / "best_model_progressive.pth"
        if cand1.exists():
            ckpt_path = cand1
        elif cand2.exists():
            ckpt_path = cand2
        else:
            ckpt_path = cand3
    else:
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_absolute():
            ckpt_path = PROJECT_ROOT / ckpt_path
            
    print(f"\n[INFO] Loading model from: {ckpt_path.relative_to(PROJECT_ROOT)}")
    model, config = load_model(ckpt_path, device)
    image_size = config.get("image_size", 224)
    
    # Load and transform image
    raw_bgr = cv2.imread(str(img_path))
    if raw_bgr is None:
        raise ValueError(f"Could not decode image at {img_path}")
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    
    val_transform = get_val_transforms(image_size=image_size)
    augmented = val_transform(image=raw_rgb)
    input_tensor = augmented["image"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs["type_logits"], dim=-1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_class = IDX_TO_CLASS[pred_idx]
        confidence = probs[pred_idx].item()
        
        # Severity predictions
        sev_scalar = outputs["severity_scalar"][0].item()
        ord_probs = torch.sigmoid(outputs["ordinal_logits"])[0].cpu().numpy()
        ord_discrete = 1 + int(np.sum(ord_probs > 0.5))
        
    # Output formatting
    proto = SEVERITY_PROTOCOLS.get(ord_discrete, SEVERITY_PROTOCOLS[1])
    
    print("\n" + "=" * 65)
    print("       WIND TURBINE BLADE DEFECT DIAGNOSTIC ASSESSMENT")
    print("=" * 65)
    print(f"• Input Image      : {img_path.name} (Source: {img_path.parent.name})")
    print(f"• Target Dimension : {raw_bgr.shape[1]}x{raw_bgr.shape[0]} RGB")
    print(f"• Inferred Defect  : {pred_class.upper()} ({pred_class})")
    print(f"• Class Confidence : {confidence * 100:.2f}%")
    print(f"• Severity Level   : LEVEL {ord_discrete} (Continuous Score: {sev_scalar:.2f} / 5.0)")
    print(f"• Severity Urgency : {proto['urgency']}")
    print(f"• Ordinal Probabilities:")
    for k in range(1, 5):
        print(f"    P(Severity > Level {k}) = {ord_probs[k-1]*100:.1f}%")
    print(f"• Action Protocol  : {proto['protocol']}")
    print("=" * 65)
    
    # Save annotated overlay
    out_dir = PROJECT_ROOT / "outputs" / "predictions"
    out_filename = f"prediction_{img_path.stem}.jpg"
    out_path = out_dir / out_filename
    draw_annotation_overlay(raw_rgb, pred_class, confidence, ord_discrete, sev_scalar, out_path)
    print(f"[SUCCESS] Diagnostic visual overlay saved to: {out_path.relative_to(PROJECT_ROOT)}\n")
    return {
        "image": img_path.name,
        "pred_class": pred_class,
        "confidence": confidence,
        "severity_level": ord_discrete,
        "severity_scalar": sev_scalar,
        "overlay_path": str(out_path)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict wind turbine blade damage and severity.")
    parser.add_argument("--image", type=str, required=True, help="Path to input inspection image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    args = parser.parse_args()
    
    run_inference(args.image, args.checkpoint)
