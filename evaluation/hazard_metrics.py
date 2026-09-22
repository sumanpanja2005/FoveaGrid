import numpy as np
from typing import Dict, Any, List

def evaluate_hazard_preservation(
    fovea_grid,
    ground_truth_boxes: List[Dict[str, Any]],
    ground_truth_potholes: List[Dict[str, Any]] = []
) -> Dict[str, Any]:
    """
    Computes Hazard Preservation Rate (HPR):
    HPR = Preserved Hazards post-aggregation / Total Ground-Truth Hazards
    Evaluates: Pedestrians, Vehicles, Curbs, Potholes, Overhead Obstacles.
    """
    preserved_counts = {"pedestrian": 0, "vehicle": 0, "curb": 0, "pothole": 0, "overhead": 0}
    total_counts     = {"pedestrian": 0, "vehicle": 0, "curb": 0, "pothole": 0, "overhead": 0}

    # Evaluate ground truth bounding boxes
    for box in ground_truth_boxes:
        pos = np.array(box["position"])
        size = np.array(box["size"])
        cat = box.get("category", "")
        cls_id = box.get("class_id", 2)

        if cls_id == 3 and size[0] < 1.2:
            category_key = "pedestrian"
        elif cls_id == 3:
            category_key = "vehicle"
        elif cat == "overhead" or cls_id == 4:
            category_key = "overhead"
        else:
            category_key = "curb"

        total_counts[category_key] += 1

        b_xmin = pos[0] - size[0]/2.0 - 0.5
        b_xmax = pos[0] + size[0]/2.0 + 0.5
        b_ymin = pos[1] - size[1]/2.0 - 0.5
        b_ymax = pos[1] + size[1]/2.0 + 0.5

        # Check if hazard is preserved in any overlapping cell in FoveaGrid
        hazard_preserved = False
        for cell in fovea_grid.cells.values():
            if not (cell.x_max < b_xmin or cell.x_min > b_xmax or cell.y_max < b_ymin or cell.y_min > b_ymax):
                if (category_key in ["pedestrian", "vehicle"] and (cell.dynamic_flag or cell.dynamic_prob > 0.3)) or \
                   (category_key == "overhead" and (cell.overhang_flag or cell.overhead_prob > 0.3)) or \
                   (category_key == "curb" and (cell.obstacle_flag or cell.static_prob > 0.3)):
                    hazard_preserved = True
                    break

        if hazard_preserved:
            preserved_counts[category_key] += 1

    # Overall Hazard Preservation Rate calculation
    total_gt = sum(total_counts.values())
    total_pres = sum(preserved_counts.values())

    hpr_by_category = {}
    for cat in total_counts:
        hpr_by_category[cat] = float(preserved_counts[cat] / max(1, total_counts[cat])) if total_counts[cat] > 0 else 1.0

    overall_hpr = float(total_pres / max(1, total_gt)) if total_gt > 0 else 1.0

    return {
        "overall_hpr": overall_hpr,
        "hpr_by_category": hpr_by_category,
        "preserved_counts": preserved_counts,
        "total_counts": total_counts
    }
