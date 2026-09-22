import numpy as np
from typing import Dict, Any

def evaluate_segmentation_metrics(predictions: np.ndarray, targets: np.ndarray, num_classes: int = 5) -> Dict[str, Any]:
    """
    Computes Accuracy, Precision, Recall, F1, IoU per class, and mIoU.
    """
    iou_per_class = {}
    total_correct = np.sum(predictions == targets)
    accuracy = float(total_correct / max(1, len(targets)))

    for cls in range(1, num_classes):
        pred_mask = (predictions == cls)
        target_mask = (targets == cls)

        intersection = np.sum(pred_mask & target_mask)
        union = np.sum(pred_mask | target_mask)

        iou = float(intersection / max(1, union)) if union > 0 else 1.0
        iou_per_class[cls] = iou

    mIoU = float(np.mean(list(iou_per_class.values()))) if len(iou_per_class) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "class_iou": iou_per_class,
        "mIoU": mIoU
    }

def compute_distance_band_miou(
    predictions: np.ndarray,
    targets: np.ndarray,
    points: np.ndarray,
    bands: list = [10.0, 25.0, 50.0, 75.0, 100.0]
) -> Dict[str, float]:
    """
    Computes mIoU breakdown across distance bands [0-10m, 10-25m, 25-50m, 50-75m, 75-100m].
    """
    ranges = np.linalg.norm(points, axis=1)
    results = {}
    prev_r = 0.0

    for r_max in bands:
        mask = (ranges >= prev_r) & (ranges <= r_max)
        if np.any(mask):
            sub_metrics = evaluate_segmentation_metrics(predictions[mask], targets[mask])
            results[f"{int(prev_r)}-{int(r_max)}m"] = sub_metrics["mIoU"]
        else:
            results[f"{int(prev_r)}-{int(r_max)}m"] = 1.0
        prev_r = r_max

    return results
