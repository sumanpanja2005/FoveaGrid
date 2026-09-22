import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter

@dataclass
class DEMGrid:
    """
    2.5D Digital Elevation Model (DEM) Raster Grid:
    Surface height Z is represented as a single-valued function Z = f(X, Y).
    """
    elevation: np.ndarray         # (H, W) full 2.5D height surface (ground + obstacle peaks)
    ground_elevation: np.ndarray  # (H, W) base terrain elevation surface
    obstacle_height: np.ndarray   # (H, W) relative obstacle height above terrain (Z - ground_Z)
    mask_observed: np.ndarray     # (H, W) boolean mask of directly sampled LiDAR cells
    x_coords: np.ndarray          # (W,) 1D coordinate axis
    y_coords: np.ndarray          # (H,) 1D coordinate axis
    mesh_x: np.ndarray            # (H, W) 2D meshgrid for 3D surface rendering
    mesh_y: np.ndarray            # (H, W) 2D meshgrid for 3D surface rendering
    bounds: Tuple[float, float, float, float]
    resolution: float

class DEMGenerator:
    """
    Constructs a 2.5D Digital Elevation Model (DEM) from classified 3D LiDAR point clouds.
    Includes hole filling (k-NN / Inverse Distance Weighting) and risk-preserving peak height retention.
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float] = (-40.0, 40.0, -40.0, 40.0),
        resolution: float = 0.25,
        inpaint_holes: bool = True
    ):
        self.bounds = bounds
        self.resolution = resolution
        self.inpaint_holes = inpaint_holes

        self.width = int(np.round((bounds[1] - bounds[0]) / resolution))
        self.height = int(np.round((bounds[3] - bounds[2]) / resolution))

        self.x_axis = np.linspace(bounds[0] + resolution / 2.0, bounds[1] - resolution / 2.0, self.width)
        self.y_axis = np.linspace(bounds[2] + resolution / 2.0, bounds[3] - resolution / 2.0, self.height)
        self.mesh_x, self.mesh_y = np.meshgrid(self.x_axis, self.y_axis)

    def generate(
        self,
        points: np.ndarray,
        ground_mask: np.ndarray
    ) -> DEMGrid:
        """
        Builds the 2.5D DEM raster grid.
        """
        total_cells = self.width * self.height
        raw_z = np.full(total_cells, np.nan, dtype=np.float32)
        ground_z = np.full(total_cells, np.nan, dtype=np.float32)

        if len(points) > 0:
            x, y, z = points[:, 0], points[:, 1], points[:, 2]

            in_b = (
                (x >= self.bounds[0]) & (x < self.bounds[1]) &
                (y >= self.bounds[2]) & (y < self.bounds[3])
            )

            x_b = x[in_b]
            y_b = y[in_b]
            z_b = z[in_b]
            is_g = ground_mask[in_b]

            gx = np.clip(np.floor((x_b - self.bounds[0]) / self.resolution).astype(int), 0, self.width - 1)
            gy = np.clip(np.floor((y_b - self.bounds[2]) / self.resolution).astype(int), 0, self.height - 1)

            flat_idx = gy * self.width + gx

            # 1. Full 2.5D surface: max peak per cell (obstacles prevail over ground)
            order = np.argsort(flat_idx)
            sorted_flat = flat_idx[order]
            sorted_z = z_b[order]
            u_bins, split_idx = np.unique(sorted_flat, return_index=True)
            max_peaks = np.maximum.reduceat(sorted_z, split_idx)
            raw_z[u_bins] = max_peaks

            # 2. Ground base surface: median/min ground points
            if np.any(is_g):
                g_flat = flat_idx[is_g]
                g_z = z_b[is_g]
                g_order = np.argsort(g_flat)
                g_sorted_flat = g_flat[g_order]
                g_sorted_z = g_z[g_order]
                g_u_bins, g_split_idx = np.unique(g_sorted_flat, return_index=True)
                g_mins = np.minimum.reduceat(g_sorted_z, g_split_idx)
                ground_z[g_u_bins] = g_mins

        elevation_2d = raw_z.reshape((self.height, self.width))
        ground_2d = ground_z.reshape((self.height, self.width))
        mask_observed = ~np.isnan(elevation_2d)

        # Inpaint unobserved ground cells via k-NN distance interpolation
        if self.inpaint_holes and np.any(np.isnan(ground_2d)):
            ground_2d = self._inpaint_surface(ground_2d)

        if self.inpaint_holes and np.any(np.isnan(elevation_2d)):
            # Unobserved areas take ground baseline height
            elevation_2d = np.where(np.isnan(elevation_2d), ground_2d, elevation_2d)

        # Ensure ground baseline doesn't exceed elevation peak
        ground_2d = np.minimum(ground_2d, elevation_2d)
        obstacle_height = np.maximum(0.0, elevation_2d - ground_2d)

        return DEMGrid(
            elevation=elevation_2d,
            ground_elevation=ground_2d,
            obstacle_height=obstacle_height,
            mask_observed=mask_observed,
            x_coords=self.x_axis,
            y_coords=self.y_axis,
            mesh_x=self.mesh_x,
            mesh_y=self.mesh_y,
            bounds=self.bounds,
            resolution=self.resolution
        )

    def _inpaint_surface(self, surface: np.ndarray) -> np.ndarray:
        """Fills missing raster values using nearest-neighbor / KD-Tree inpainting."""
        nan_mask = np.isnan(surface)
        valid_mask = ~nan_mask

        if not np.any(valid_mask):
            return np.zeros_like(surface)

        valid_idx = np.column_stack(np.where(valid_mask))
        nan_idx = np.column_stack(np.where(nan_mask))

        if len(nan_idx) == 0:
            return surface

        tree = cKDTree(valid_idx)
        _, nearest = tree.query(nan_idx, k=1)

        filled = surface.copy()
        filled[nan_mask] = surface[valid_idx[nearest, 0], valid_idx[nearest, 1]]

        # Light smoothing filter on filled ground surface
        filled = gaussian_filter(filled, sigma=1.0)
        # Preserve original observed cells
        filled[valid_mask] = surface[valid_mask]
        return filled
