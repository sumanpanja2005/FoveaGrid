import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class BEVGrid:
    """
    Multi-Channel 2D Bird's Eye View (BEV) Grid Representation:
    Preserves top-down spatial occupancy along with 3D elevation and intensity metadata.
    """
    occupancy: np.ndarray   # (H, W) [0..1]
    z_max: np.ndarray       # (H, W) max elevation
    z_min: np.ndarray       # (H, W) min ground elevation
    z_delta: np.ndarray     # (H, W) height span (z_max - z_min)
    intensity: np.ndarray   # (H, W) mean intensity
    density: np.ndarray     # (H, W) point count
    bounds: Tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)
    resolution: float

    @property
    def tensor(self) -> np.ndarray:
        """Returns 6-channel stacked tensor: (H, W, 6)."""
        return np.stack([
            self.occupancy,
            self.z_max,
            self.z_min,
            self.z_delta,
            self.intensity,
            self.density
        ], axis=-1)

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        gx = int(np.floor((wx - self.bounds[0]) / self.resolution))
        gy = int(np.floor((wy - self.bounds[2]) / self.resolution))
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        wx = self.bounds[0] + (gx + 0.5) * self.resolution
        wy = self.bounds[2] + (gy + 0.5) * self.resolution
        return wx, wy

class BEVProjector:
    """
    Fast Vectorized 2D BEV Grid Generator from 3D LiDAR Point Clouds.
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float] = (-40.0, 40.0, -40.0, 40.0),
        resolution: float = 0.20
    ):
        self.bounds = bounds
        self.resolution = resolution
        self.width = int(np.round((bounds[1] - bounds[0]) / resolution))
        self.height = int(np.round((bounds[3] - bounds[2]) / resolution))

    def project(
        self,
        points: np.ndarray,
        ground_mask: Optional[np.ndarray] = None,
        intensities: Optional[np.ndarray] = None
    ) -> BEVGrid:
        """
        Projects point cloud onto a regular 2D BEV grid with height and intensity channels.
        """
        if len(points) == 0:
            zeros = np.zeros((self.height, self.width), dtype=np.float32)
            return BEVGrid(
                occupancy=zeros, z_max=zeros, z_min=zeros, z_delta=zeros,
                intensity=zeros, density=zeros, bounds=self.bounds, resolution=self.resolution
            )

        if intensities is None:
            intensities = np.zeros(len(points), dtype=np.float32)
        if ground_mask is None:
            ground_mask = np.zeros(len(points), dtype=bool)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Filter within bounds
        in_bounds = (
            (x >= self.bounds[0]) & (x < self.bounds[1]) &
            (y >= self.bounds[2]) & (y < self.bounds[3])
        )

        pts_valid = points[in_bounds]
        z_valid = z[in_bounds]
        inten_valid = intensities[in_bounds]
        is_ground = ground_mask[in_bounds]

        gx = np.floor((pts_valid[:, 0] - self.bounds[0]) / self.resolution).astype(int)
        gy = np.floor((pts_valid[:, 1] - self.bounds[2]) / self.resolution).astype(int)

        gx = np.clip(gx, 0, self.width - 1)
        gy = np.clip(gy, 0, self.height - 1)

        # 1D flattened bin index for vectorization
        flat_idx = gy * self.width + gx
        total_cells = self.width * self.height

        # Initialize layers
        occupancy = np.zeros(total_cells, dtype=np.float32)
        z_max = np.full(total_cells, -999.0, dtype=np.float32)
        z_min = np.full(total_cells, 999.0, dtype=np.float32)
        intensity_sum = np.zeros(total_cells, dtype=np.float32)
        density = np.zeros(total_cells, dtype=np.float32)

        # Density count per cell
        counts = np.bincount(flat_idx, minlength=total_cells)
        density = counts.astype(np.float32)
        valid_cells = density > 0

        # Sort by flat_idx for reduceat
        order = np.argsort(flat_idx)
        sorted_flat = flat_idx[order]
        sorted_z = z_valid[order]
        sorted_inten = inten_valid[order]
        sorted_obs = (~is_ground)[order]

        unique_bins, split_idx = np.unique(sorted_flat, return_index=True)

        # Max & min elevation
        max_z_bins = np.maximum.reduceat(sorted_z, split_idx)
        min_z_bins = np.minimum.reduceat(sorted_z, split_idx)
        sum_inten_bins = np.add.reduceat(sorted_inten, split_idx)
        obs_bins = np.maximum.reduceat(sorted_obs.astype(np.float32), split_idx)

        z_max[unique_bins] = max_z_bins
        z_min[unique_bins] = min_z_bins
        occupancy[unique_bins] = obs_bins
        intensity_sum[unique_bins] = sum_inten_bins

        # Normalize intensity
        mean_intensity = np.zeros(total_cells, dtype=np.float32)
        mean_intensity[valid_cells] = intensity_sum[valid_cells] / density[valid_cells]

        # Reset unobserved cells to 0.0
        z_max[~valid_cells] = 0.0
        z_min[~valid_cells] = 0.0

        # Height span
        z_delta = np.maximum(0.0, z_max - z_min)
        z_delta[~valid_cells] = 0.0

        # Reshape to (H, W)
        return BEVGrid(
            occupancy=occupancy.reshape((self.height, self.width)),
            z_max=z_max.reshape((self.height, self.width)),
            z_min=z_min.reshape((self.height, self.width)),
            z_delta=z_delta.reshape((self.height, self.width)),
            intensity=mean_intensity.reshape((self.height, self.width)),
            density=density.reshape((self.height, self.width)),
            bounds=self.bounds,
            resolution=self.resolution
        )
