import numpy as np
from typing import List
from .cell import FoveaCell

def risk_preserving_pool(child_cells: List[FoveaCell], target_resolution: float) -> FoveaCell:
    """
    CRITICAL RISK-PRESERVING CONSERVATIVE POOLING:
    Merges fine child cells into a single coarse parent cell WITHOUT losing safety-critical hazards.

    Rules:
    - parent.obstacle_flag = ANY(child.obstacle_flag)
    - parent.dynamic_flag  = ANY(child.dynamic_flag)
    - parent.overhang_flag = ANY(child.overhang_flag)
    - parent.ground_min    = MIN(child.ground_min)
    - parent.ground_max    = MAX(child.ground_max)
    - parent.obstacle_max  = MAX(child.obstacle_max)
    """
    if len(child_cells) == 0:
        return FoveaCell(resolution=target_resolution)

    # Compute bounding region
    x_min = min(c.x_min for c in child_cells)
    x_max = max(c.x_max for c in child_cells)
    y_min = min(c.y_min for c in child_cells)
    y_max = max(c.y_max for c in child_cells)

    parent = FoveaCell(
        x_min=x_min, x_max=x_max,
        y_min=y_min, y_max=y_max,
        resolution=target_resolution
    )

    # 1. Conservative Risk Flags: ANY obstacle/dynamic/overhang turns parent flag ON
    parent.obstacle_flag = any(c.obstacle_flag for c in child_cells)
    parent.dynamic_flag  = any(c.dynamic_flag  for c in child_cells)
    parent.overhang_flag = any(c.overhang_flag for c in child_cells)

    # 2. Conservative Height Bounds: Min/Max preservation
    parent.ground_height_min = min(c.ground_height_min for c in child_cells)
    parent.ground_height_max = max(c.ground_height_max for c in child_cells)
    
    parent.obstacle_height_min = min(c.obstacle_height_min for c in child_cells)
    parent.obstacle_height_max = max(c.obstacle_height_max for c in child_cells)

    # 3. Maximum-Preservation Probability Aggregation
    # Risk-preserving: peak hazardous probability is preserved rather than diluted
    parent.static_prob   = max(c.static_prob   for c in child_cells)
    parent.dynamic_prob  = max(c.dynamic_prob  for c in child_cells)
    parent.overhead_prob = max(c.overhead_prob for c in child_cells)
    parent.terrain_prob  = min(c.terrain_prob  for c in child_cells) if parent.obstacle_flag else np.mean([c.terrain_prob for c in child_cells])

    # Normalize semantic probabilities
    sum_p = parent.terrain_prob + parent.static_prob + parent.dynamic_prob + parent.overhead_prob
    if sum_p > 1e-6:
        parent.terrain_prob /= sum_p
        parent.static_prob  /= sum_p
        parent.dynamic_prob /= sum_p
        parent.overhead_prob /= sum_p

    # 4. Max Velocity Vector Aggregation
    max_dyn_cell = max(child_cells, key=lambda c: np.hypot(c.velocity_x, c.velocity_y))
    parent.velocity_x = max_dyn_cell.velocity_x
    parent.velocity_y = max_dyn_cell.velocity_y

    # 5. Observation count & timestamp
    parent.observation_count = sum(c.observation_count for c in child_cells)
    parent.last_update_timestamp = max(c.last_update_timestamp for c in child_cells)

    # 6. Preserved Confidence Calculation
    parent.confidence = np.mean([c.confidence for c in child_cells])

    return parent
