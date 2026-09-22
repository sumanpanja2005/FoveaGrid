import numpy as np
from typing import Dict, List, Tuple, Any
from .cell import FoveaCell
from .pooling import risk_preserving_pool
from .ring_buffer import RingBufferIndex
from .confidence import compute_cell_confidence
from .fusion import TemporalProbabilisticFusion
from .micro_fovea import MicroFoveaAllocator

class FoveaGrid:
    """
    Adaptive Variable-Resolution 2.5D Foveated LiDAR World Mapper:
    - High spatial resolution near vehicle (5 cm within 10 m)
    - Progressive coarse resolution farther away (growing to 50 cm at 100 m)
    - Risk-Preserving Aggregation (Obstacle-wins pooling, max dynamic/overhang flags, min/max heights)
    - Speed-steered fovea forward elongation
    - Object-based micro-foveas with fixed memory cell budget
    - Toroidal ring-buffer world translation
    """

    def __init__(
        self,
        fovea_bands: List[Dict[str, float]] = None,
        world_size: float = 200.0,
        base_resolution: float = 0.05
    ):
        self.world_size = world_size
        self.base_resolution = base_resolution

        self.fovea_bands = fovea_bands or [
            {"range": 10.0,  "resolution": 0.05},
            {"range": 25.0,  "resolution": 0.10},
            {"range": 50.0,  "resolution": 0.20},
            {"range": 75.0,  "resolution": 0.35},
            {"range": 100.0, "resolution": 0.50},
        ]

        self.micro_allocator = MicroFoveaAllocator()
        self.fusion_engine = TemporalProbabilisticFusion()
        
        self.dim_base = int(np.round(world_size / base_resolution))
        self.ring_buffer = RingBufferIndex(self.dim_base, self.dim_base, cell_size=base_resolution)
        self.cells: Dict[Tuple[int, int, float], FoveaCell] = {}

    def get_desired_resolution(self, wx: float, wy: float, ego_pos: np.ndarray, ego_vel: np.ndarray) -> float:
        if self.micro_allocator.is_in_micro_fovea(wx, wy):
            return 0.05

        speed = np.hypot(ego_vel[0], ego_vel[1])
        if speed > 2.0:
            heading = np.arctan2(ego_vel[1], ego_vel[0])
            forward_shift = min(25.0, speed * 1.0)
            fovea_center_x = ego_pos[0] + forward_shift * np.cos(heading)
            fovea_center_y = ego_pos[1] + forward_shift * np.sin(heading)
        else:
            fovea_center_x = ego_pos[0]
            fovea_center_y = ego_pos[1]

        dist = np.hypot(wx - fovea_center_x, wy - fovea_center_y)
        for band in self.fovea_bands:
            if dist <= band["range"]:
                return band["resolution"]

        return 0.50

    def update(
        self,
        points: np.ndarray,
        semantic_probs: np.ndarray,
        dynamic_probs: np.ndarray,
        ego_pos: np.ndarray,
        ego_vel: np.ndarray,
        tracked_targets: List[Dict[str, Any]] = [],
        timestamp: float = 0.0
    ):
        self.micro_allocator.update_micro_foveas(tracked_targets, ego_pos)
        self.ring_buffer.shift_origin(ego_pos[0], ego_pos[1])

        if len(points) == 0:
            return

        x_pts, y_pts, z_pts = points[:, 0], points[:, 1], points[:, 2]

        speed = np.hypot(ego_vel[0], ego_vel[1])
        if speed > 2.0:
            heading = np.arctan2(ego_vel[1], ego_vel[0])
            shift = min(25.0, speed * 1.0)
            fc_x = ego_pos[0] + shift * np.cos(heading)
            fc_y = ego_pos[1] + shift * np.sin(heading)
        else:
            fc_x, fc_y = ego_pos[0], ego_pos[1]

        dists = np.hypot(x_pts - fc_x, y_pts - fc_y)
        
        target_res = np.full(len(points), 0.50, dtype=np.float32)
        target_res[dists <= 75.0] = 0.35
        target_res[dists <= 50.0] = 0.20
        target_res[dists <= 25.0] = 0.10
        target_res[dists <= 10.0] = 0.05

        for mf in self.micro_allocator.active_micro_foveas:
            in_mf = np.hypot(x_pts - mf.center_x, y_pts - mf.center_y) <= mf.radius
            target_res[in_mf] = 0.05

        ix = np.floor(x_pts / target_res).astype(int)
        iy = np.floor(y_pts / target_res).astype(int)
        res_key = np.round(target_res * 100).astype(int)

        keys = np.stack([ix, iy, res_key], axis=1)
        unique_keys, inverse_indices = np.unique(keys, axis=0, return_inverse=True)

        # O(N) grouping speed optimization
        sort_order = np.argsort(inverse_indices)
        inv_sorted = inverse_indices[sort_order]
        split_idx = np.where(np.diff(inv_sorted) != 0)[0] + 1
        grouped_indices = np.split(sort_order, split_idx)

        new_grid: Dict[Tuple[int, int, float], FoveaCell] = {}

        for idx, ukey in enumerate(unique_keys):
            mask = grouped_indices[idx]
            pt_x, pt_y, pt_z = x_pts[mask], y_pts[mask], z_pts[mask]
            p_sem = semantic_probs[mask]
            p_dyn = dynamic_probs[mask]

            cell_res = float(ukey[2]) / 100.0
            cell_wx = (ukey[0] + 0.5) * cell_res
            cell_wy = (ukey[1] + 0.5) * cell_res

            g_min, g_max = float(np.min(pt_z)), float(np.mean(pt_z))
            obs_mask = p_sem[:, 1] < 0.5
            obs_max = float(np.max(pt_z[obs_mask])) if np.any(obs_mask) else g_max
            obs_min = float(np.min(pt_z[obs_mask])) if np.any(obs_mask) else g_max

            obs_flag = bool(np.any(obs_mask))
            dyn_flag = bool(np.any(p_dyn > 0.4))
            overhang_flag = bool(np.any(p_sem[:, 4] > 0.4))

            mean_sem = np.mean(p_sem, axis=0)

            cell = FoveaCell(
                x_min=cell_wx - cell_res/2.0, x_max=cell_wx + cell_res/2.0,
                y_min=cell_wy - cell_res/2.0, y_max=cell_wy + cell_res/2.0,
                resolution=cell_res,
                ground_height_min=g_min, ground_height_max=g_max,
                obstacle_height_min=obs_min, obstacle_height_max=obs_max,
                terrain_prob=float(mean_sem[1]), static_prob=float(mean_sem[2]),
                dynamic_prob=float(max(mean_sem[3], np.max(p_dyn))), overhead_prob=float(mean_sem[4]),
                obstacle_flag=obs_flag, dynamic_flag=dyn_flag, overhang_flag=overhang_flag,
                observation_count=len(mask), last_update_timestamp=timestamp
            )

            dist_ego = np.hypot(cell_wx - ego_pos[0], cell_wy - ego_pos[1])
            cell.confidence = compute_cell_confidence(cell.observation_count, dist_ego)

            dict_key = (ukey[0], ukey[1], cell_res)
            new_grid[dict_key] = cell

        self.cells = new_grid

    def get_cell_count(self) -> int:
        return len(self.cells)

    def get_memory_bytes(self) -> int:
        return len(self.cells) * 128
