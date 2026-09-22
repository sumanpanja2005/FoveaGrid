import numpy as np
from typing import List, Tuple, Dict, Any

class MicroFoveaRegion:
    def __init__(self, center_x: float, center_y: float, radius: float = 3.0, target_resolution: float = 0.05):
        self.center_x = center_x
        self.center_y = center_y
        self.radius = radius
        self.target_resolution = target_resolution

class MicroFoveaAllocator:
    """
    Allocates high-resolution micro-foveas around distant dynamic objects or road hazards,
    while maintaining a strict fixed cell budget across the map grid.
    """

    def __init__(
        self,
        cell_budget: int = 300000,
        max_micro_foveas: int = 4,
        micro_fovea_radius: float = 3.0,
        high_resolution: float = 0.05
    ):
        self.cell_budget = cell_budget
        self.max_micro_foveas = max_micro_foveas
        self.micro_fovea_radius = micro_fovea_radius
        self.high_resolution = high_resolution
        self.active_micro_foveas: List[MicroFoveaRegion] = []

    def update_micro_foveas(
        self,
        tracked_targets: List[Dict[str, Any]],
        ego_pos: np.ndarray
    ) -> List[MicroFoveaRegion]:
        """
        Selects top priority distant dynamic targets/hazards for micro-fovea high-res allocation.
        """
        active = []
        for target in tracked_targets:
            pos = target["position"] if isinstance(target["position"], np.ndarray) else np.array(target["position"])
            dist = np.hypot(pos[0] - ego_pos[0], pos[1] - ego_pos[1])

            # Targets beyond central 10m fovea get micro-fovea allocation
            if 10.0 < dist <= 75.0 and len(active) < self.max_micro_foveas:
                active.append(MicroFoveaRegion(
                    center_x=pos[0],
                    center_y=pos[1],
                    radius=self.micro_fovea_radius,
                    target_resolution=self.high_resolution
                ))

        self.active_micro_foveas = active
        return self.active_micro_foveas

    def is_in_micro_fovea(self, wx: float, wy: float) -> bool:
        """Checks if world coordinate (wx, wy) falls inside any active micro-fovea region."""
        for mf in self.active_micro_foveas:
            if np.hypot(wx - mf.center_x, wy - mf.center_y) <= mf.radius:
                return True
        return False
