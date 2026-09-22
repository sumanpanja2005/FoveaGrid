import numpy as np

class GroundEstimator:
    """
    Estimates local ground surface height $z_{ground}(x, y)$ using grid-min surface projection
    and local plane fitting.
    """

    def __init__(self, grid_size: float = 1.0, height_threshold: float = 0.25):
        self.grid_size = grid_size
        self.height_threshold = height_threshold

    def estimate_ground(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
        - ground_heights: estimated ground elevation per point
        - is_ground_mask: boolean mask indicating ground points
        """
        if len(points) == 0:
            return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=bool)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]

        # Discretize into coarse 2D grid cells
        ix = np.floor(x / self.grid_size).astype(int)
        iy = np.floor(y / self.grid_size).astype(int)

        cell_keys = (ix, iy)
        unique_cells, inverse_indices = np.unique(np.stack(cell_keys, axis=1), axis=0, return_inverse=True)

        # Compute minimum z per cell
        min_z_per_cell = np.full(len(unique_cells), 1e5, dtype=np.float32)
        np.minimum.at(min_z_per_cell, inverse_indices, z)

        # Assign estimated ground height to each point based on its cell min_z
        estimated_ground_z = min_z_per_cell[inverse_indices]

        # Points close to cell minimum height are ground points
        is_ground = np.abs(z - estimated_ground_z) <= self.height_threshold

        return estimated_ground_z.astype(np.float32), is_ground
