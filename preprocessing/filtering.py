import numpy as np

class RangeFilter:
    """Removes points outside [min_range, max_range]."""
    def __init__(self, min_range: float = 0.5, max_range: float = 100.0):
        self.min_range = min_range
        self.max_range = max_range

    def filter(self, points: np.ndarray, *attrs) -> tuple:
        ranges = np.linalg.norm(points, axis=1)
        valid = (ranges >= self.min_range) & (ranges <= self.max_range)
        
        filtered_pts = points[valid]
        filtered_attrs = tuple(attr[valid] for attr in attrs if attr is not None)
        return (filtered_pts,) + filtered_attrs

class StatisticalOutlierFilter:
    """Filters noisy isolated outlier points."""
    def __init__(self, k_neighbors: int = 10, std_ratio: float = 2.0):
        self.k = k_neighbors
        self.std_ratio = std_ratio

    def filter(self, points: np.ndarray) -> np.ndarray:
        if len(points) < self.k:
            return points
        # Simple fast distance filter
        from scipy.spatial import KDTree
        tree = KDTree(points)
        dists, _ = tree.query(points, k=self.k)
        mean_dists = np.mean(dists[:, 1:], axis=1)
        thresh = np.mean(mean_dists) + self.std_ratio * np.std(mean_dists)
        return points[mean_dists <= thresh]
