import numpy as np
from typing import Dict, Tuple
from .cell import FoveaCell

class UniformBaselineGrid:
    """
    Baseline Uniform 2.5D World Map:
    Fixed 5 cm resolution everywhere across 200 m x 200 m ROI (4000 x 4000 grid).
    Used to benchmark cell count, memory consumption, and timing against FoveaGrid.
    """

    def __init__(self, world_size: float = 200.0, resolution: float = 0.05):
        self.world_size = world_size
        self.resolution = resolution
        self.grid_dim = int(np.round(world_size / resolution)) # 4000
        self.cells: Dict[Tuple[int, int], FoveaCell] = {}

    def update(
        self,
        points: np.ndarray,
        semantic_probs: np.ndarray,
        dynamic_probs: np.ndarray,
        ego_pos: np.ndarray,
        timestamp: float = 0.0
    ):
        if len(points) == 0:
            return

        x_pts, y_pts, z_pts = points[:, 0], points[:, 1], points[:, 2]

        ix_grid = np.floor((x_pts + self.world_size / 2.0) / self.resolution).astype(int)
        iy_grid = np.floor((y_pts + self.world_size / 2.0) / self.resolution).astype(int)

        keys = np.stack([ix_grid, iy_grid], axis=1)
        unique_keys, inverse_indices = np.unique(keys, axis=0, return_inverse=True)

        # O(N) grouping using argsort & split
        sort_order = np.argsort(inverse_indices)
        inv_sorted = inverse_indices[sort_order]
        
        # Find split boundaries where index changes
        split_idx = np.where(np.diff(inv_sorted) != 0)[0] + 1
        grouped_indices = np.split(sort_order, split_idx)

        new_cells = {}
        for idx, ukey in enumerate(unique_keys):
            mask = grouped_indices[idx]
            pt_x = x_pts[mask]
            pt_y = y_pts[mask]
            pt_z = z_pts[mask]

            p_sem = semantic_probs[mask]
            p_dyn = dynamic_probs[mask]

            wx = (ukey[0] + 0.5) * self.resolution - self.world_size / 2.0
            wy = (ukey[1] + 0.5) * self.resolution - self.world_size / 2.0

            g_min, g_max = float(np.min(pt_z)), float(np.mean(pt_z))
            obs_mask = p_sem[:, 1] < 0.5
            obs_max = float(np.max(pt_z[obs_mask])) if np.any(obs_mask) else g_max

            cell = FoveaCell(
                x_min=wx - 0.025, x_max=wx + 0.025,
                y_min=wy - 0.025, y_max=wy + 0.025,
                resolution=0.05,
                ground_height_min=g_min, ground_height_max=g_max,
                obstacle_height_max=obs_max,
                obstacle_flag=bool(np.any(obs_mask)),
                dynamic_flag=bool(np.any(p_dyn > 0.4)),
                overhang_flag=bool(np.any(p_sem[:, 4] > 0.4)),
                observation_count=len(mask),
                last_update_timestamp=timestamp
            )

            new_cells[(ukey[0], ukey[1])] = cell

        self.cells = new_cells

    def get_cell_count(self) -> int:
        return len(self.cells)

    def get_memory_bytes(self) -> int:
        return len(self.cells) * 128
