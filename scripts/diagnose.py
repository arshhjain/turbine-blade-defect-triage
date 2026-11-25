"""
Actionable Diagnostic CLI for Single/Batch Drone Inspection Imagery.

Runs automated classification, generates Grad-CAM explainability heatmaps,
creates side-by-side [Input Drone Frame | Model Output] comparison views,
and formats immediate, actionable maintenance recommendations.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import yaml
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.blade_classifier import TurbineBladeDefectClassifier, build_model
from inference.diagnostic_engine import BladeDiagnosticEngine


def format_cli_report(record: dict) -> str:
    """Format single diagnostic record for terminal output."""
    div = "=" * 70
    gt_classes = record.get("ground_truth_classes", [])
    gt_raw = record.get("ground_truth_raw", [])
    gt_str = f"{', '.join(gt_classes).upper()} (Raw: {', '.join(gt_raw)})" if gt_classes else "UNANNOTATED"

    verdict = record.get("verdict_match")
    if verdict is True:
        verdict_str = "MATCH [CORRECT AI DETECTION]"
    elif verdict is False:
        verdict_str = "MISMATCH [INSPECTOR REVIEW REQUIRED]"
    else:
        verdict_str = "UNANNOTATED UAV FRAME"

    return f"""
{div}
  WIND TURBINE BLADE INSPECTION REPORT: {record.get('image_filename')}
{div}
• Ground Truth Defect:  {gt_str}
• Model Prediction:     {record.get('primary_defect', '').upper()} ({record.get('confidence_percentage')})
• Detection Verdict:    {verdict_str}
• Defect Severity:      LEVEL {record.get('severity_level')} ({record.get('urgency')})
• Anomaly Description:  {record.get('anomaly_title')}
• Failure Mechanism:    {record.get('risk_mechanism')}
• Immediate Protocol:   {record.get('actionable_protocol')}
• Side-by-Side Image:   {record.get('comparison_saved_at')}
• Grad-CAM Overlay:     {record.get('cam_overlay_saved_at')}
{div}
Class Probabilities:
{json.dumps(record.get('all_class_probabilities', {}), indent=4)}
{div}
"""


def main():
    parser = argparse.ArgumentParser(description="Run Actionable Diagnosis on Wind Turbine Drone Imagery")
    parser.add_argument("--image", type=str, default=None, help="Path to single drone image")
    parser.add_argument("--image-id", type=int, default=None, help="Image index from Figshare dataset (0-1064)")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory containing drone images")
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/best_model_progressive.pth", help="Model checkpoint path")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--resolution", type=int, default=512, help="Input resolution")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--no-cam", action="store_true", help="Disable Grad-CAM heatmap generation")
    parser.add_argument("--no-compare", action="store_true", help="Disable side-by-side comparison image")
    parser.add_argument("--show", action="store_true", help="Automatically open inspection image viewer")
    args = parser.parse_args()

    # Find checkpoint or fallback
    ckpt_path = PROJECT_ROOT / args.checkpoint
    config_path = PROJECT_ROOT / args.config

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device_str = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    # Initialize model
    model = build_model(config).to(device)

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        class_to_idx = checkpoint.get("class_to_idx", {cls: i for i, cls in enumerate(config["dataset"]["classes"])})
        print(f"Loaded trained checkpoint from: {ckpt_path}")
    else:
        print(f"[NOTE] Checkpoint {ckpt_path} not found. Running with pre-trained backbone features for demonstration.")
        class_to_idx = {cls: i for i, cls in enumerate(config["dataset"]["classes"])}

    # Engine
    engine = BladeDiagnosticEngine(
        model=model,
        class_to_idx=class_to_idx,
        device=device,
        image_size=args.resolution,
        output_dir="outputs/diagnostics"
    )

    # Identify image targets
    targets = []
    if args.image_id is not None:
        figshare_img = PROJECT_ROOT / "data/figshare/WT blade defect dataset/JPEGImages" / f"{args.image_id}.jpg"
        if figshare_img.exists():
            targets.append(str(figshare_img))
        else:
            print(f"[Error] Image index #{args.image_id} not found at {figshare_img}")
            return
    elif args.image:
        targets.append(str(PROJECT_ROOT / args.image) if not Path(args.image).is_absolute() else args.image)
    elif args.input_dir:
        in_dir = PROJECT_ROOT / args.input_dir if not Path(args.input_dir).is_absolute() else Path(args.input_dir)
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
            targets.extend([str(p) for p in in_dir.glob(ext)])
    else:
        # Check if sample Figshare image exists
        sample_img = PROJECT_ROOT / "data/figshare/WT blade defect dataset/JPEGImages/104.jpg"
        if sample_img.exists():
            targets.append(str(sample_img))
        else:
            print("No image provided. Please specify --image <path>, --image-id <0-1064>, or --input-dir <dir>.")
            return

    print(f"\nProcessing {len(targets)} wind turbine drone inspection image(s)...")
    for target in targets:
        record = engine.diagnose_image(
            target,
            generate_cam=not args.no_cam,
            generate_comparison=not args.no_compare
        )
        print(format_cli_report(record))

        if args.show:
            to_open = record.get("comparison_saved_at") or record.get("cam_overlay_saved_at")
            if to_open and Path(to_open).exists() and sys.platform.startswith("win"):
                os.system(f'start "" "{to_open}"')


if __name__ == "__main__":
    main()
