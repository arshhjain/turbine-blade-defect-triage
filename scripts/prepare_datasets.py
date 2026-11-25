"""
Dataset Integration & Ingestion Script for Wind Turbine Blade Defect Classification.

Integrates:
1. Figshare: "Multiclass Dataset for Intelligent Detection of Wind Turbine Blade Defects Using Drone Imagery"
   (Figshare DOI: 10.6084/m9.figshare.14728509 or related UAV collections)
2. Roboflow: Pre-annotated Wind Turbine Blade Object Detection datasets (YOLO format)
3. Synthetic procedural drone dataset generator for immediate 2-day MVP testing.
"""

import os
import sys
import argparse
import urllib.request
import zipfile
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets.mock_generator import generate_synthetic_turbine_dataset


FIGSHARE_INFO = """
--------------------------------------------------------------------------------
FIGSHARE DATASET INTEGRATION INSTRUCTIONS:
Dataset: "Multiclass Dataset for Intelligent Detection of Wind Turbine Blade Defects Using Drone Imagery"
URL: https://figshare.com/articles/dataset/Multiclass_Dataset_for_Intelligent_Detection_of_Wind_Turbine_Blade_Defects_Using_Drone_Imagery/14728509
Content: Over 1,000 high-resolution UAV drone inspection images capturing:
  - Surface and structural cracking
  - Leading-edge erosion
  - Delamination and composite coating breakdown
  - Normal/clean blade baseline

To integrate your downloaded Figshare archive:
1. Unpack the Figshare zip into: `data/figshare/`
2. Structure format accepted:
   data/figshare/
   ├── cracking/
   ├── erosion/
   ├── delamination/
   └── normal/
   (Or flat folder with annotation CSV metadata)
--------------------------------------------------------------------------------
"""

ROBOFLOW_INFO = """
--------------------------------------------------------------------------------
ROBOFLOW OBJECT DETECTION INTEGRATION INSTRUCTIONS:
Pre-annotated Wind Turbine Blade Defect Detection Datasets (YOLOv8 / YOLOv5 format)
Roboflow Universe: https://universe.roboflow.com/search?q=wind+turbine+blade+defects

To integrate a Roboflow export:
1. Export dataset in 'YOLOv8' or 'YOLOv5 PyTorch' format from Roboflow Universe.
2. Unpack into `data/roboflow/`
3. Structure format accepted:
   data/roboflow/
   ├── data.yaml
   ├── train/
   │   ├── images/
   │   └── labels/
   └── valid/
       ├── images/
       └── labels/
The unified dataloader automatically parses bounding boxes and extracts localized defect crops!
--------------------------------------------------------------------------------
"""


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for Wind Turbine Blade Defect Classification")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic drone benchmark dataset")
    parser.add_argument("--samples-per-class", type=int, default=30, help="Number of synthetic samples per class")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory for synthetic data")
    parser.add_argument("--show-info", action="store_true", help="Show Figshare and Roboflow download instructions")
    args = parser.parse_args()

    if args.show_info or not args.generate_synthetic:
        print(FIGSHARE_INFO)
        print(ROBOFLOW_INFO)

    if args.generate_synthetic:
        target_dir = PROJECT_ROOT / args.output_dir
        generate_synthetic_turbine_dataset(
            output_dir=str(target_dir),
            samples_per_class=args.samples_per_class,
            img_size=512,
            create_roboflow_yolo=True,
        )
        print(f"\nSynthetic dataset ready for progressive training at: {target_dir}")


if __name__ == "__main__":
    main()
