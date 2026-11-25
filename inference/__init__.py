"""
Wind Turbine Blade Defect Classification - Inference Module.
"""

from .gradcam import GradCAM
from .diagnostic_engine import BladeDiagnosticEngine, SEVERITY_CATALOG

__all__ = [
    "GradCAM",
    "BladeDiagnosticEngine",
    "SEVERITY_CATALOG",
]
