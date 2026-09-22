"""
FoveaGrid Evaluation Suite: Semantic mIoU by distance band, Hazard Preservation Rate, Memory compression ratio, Latency breakdown, and Ablations A-J.
"""

from .segmentation_metrics import evaluate_segmentation_metrics, compute_distance_band_miou
from .hazard_metrics import evaluate_hazard_preservation
from .memory_metrics import evaluate_memory_metrics
from .latency_metrics import LatencyProfiler
from .ablation import run_ablation_experiments

__all__ = [
    "evaluate_segmentation_metrics",
    "compute_distance_band_miou",
    "evaluate_hazard_preservation",
    "evaluate_memory_metrics",
    "LatencyProfiler",
    "run_ablation_experiments",
]
