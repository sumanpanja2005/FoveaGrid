import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

@dataclass
class ObstacleCluster:
    cluster_id: int
    category: str  # "vehicle", "pedestrian", "pole", "tree", "static_obstacle"
    position: np.ndarray  # [x, y, z] center
    size: np.ndarray  # [dx, dy, dz] bounding extents
    points: np.ndarray  # (K, 3) point coordinates
    linearity: float = 0.0
    verticality: float = 0.0
    confidence: float = 0.8

@dataclass
class DetectedFeatures:
    curb_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=np.float32))
    potholes: List[Dict[str, Any]] = field(default_factory=list)
    dynamic_obstacles: List[ObstacleCluster] = field(default_factory=list)  # vehicles, pedestrians
    roadside_infrastructure: List[ObstacleCluster] = field(default_factory=list)  # poles, trees
    static_obstacles: List[ObstacleCluster] = field(default_factory=list)

class RoadFeatureExtractor:
    """
    Extracts road-specific and environmental features:
    - Road boundaries & curbs (elevation steps)
    - Surface depressions / potholes
    - Clustered obstacles (vehicles, pedestrians)
    - Roadside infrastructure (poles, trees)
    """

    def __init__(
        self,
        curb_min_step: float = 0.08,
        curb_max_step: float = 0.28,
        pothole_min_depth: float = 0.08,
        cluster_eps: float = 0.7,
        cluster_min_points: int = 5
    ):
        self.curb_min_step = curb_min_step
        self.curb_max_step = curb_max_step
        self.pothole_min_depth = pothole_min_depth
        self.cluster_eps = cluster_eps
        self.cluster_min_points = cluster_min_points

    def extract_features(
        self,
        ground_points: np.ndarray,
        non_ground_points: np.ndarray,
        ground_plane: Optional[Tuple[float, float, float, float]] = None
    ) -> DetectedFeatures:
        """
        Runs comprehensive multi-class feature extraction.
        """
        features = DetectedFeatures()

        # 1. Detect Road Curbs & Boundaries
        if len(ground_points) > 50:
            features.curb_points = self.detect_curbs(ground_points)

        # 2. Detect Surface Anomalies (Potholes)
        if len(ground_points) > 50:
            features.potholes = self.detect_potholes(ground_points, ground_plane)

        # 3. Detect Obstacles & Infrastructure
        if len(non_ground_points) >= self.cluster_min_points:
            dyn_obs, infra_obs, static_obs = self.detect_and_classify_clusters(non_ground_points)
            features.dynamic_obstacles = dyn_obs
            features.roadside_infrastructure = infra_obs
            features.static_obstacles = static_obs

        return features

    def detect_curbs(self, ground_points: np.ndarray, cell_size: float = 0.25) -> np.ndarray:
        """
        Detects road curbs by finding abrupt vertical height gradients in adjacent local ground cells.
        """
        x = ground_points[:, 0]
        y = ground_points[:, 1]
        z = ground_points[:, 2]

        ix = np.floor(x / cell_size).astype(int)
        iy = np.floor(y / cell_size).astype(int)

        # Hash coordinates
        grid_keys = np.stack([ix, iy], axis=1)
        unique_keys, inverse = np.unique(grid_keys, axis=0, return_inverse=True)
        M = len(unique_keys)

        # Calculate max and min z per cell
        counts = np.bincount(inverse, minlength=M)
        valid = counts >= 2

        order = np.argsort(inverse)
        inv_sorted = inverse[order]
        split_idx = np.where(np.diff(inv_sorted) != 0)[0] + 1
        splits = np.split(order, split_idx)

        curb_point_indices = []
        for i, idx_group in enumerate(splits):
            if not valid[i]:
                continue
            z_grp = z[idx_group]
            step = float(np.max(z_grp) - np.min(z_grp))
            if self.curb_min_step <= step <= self.curb_max_step:
                curb_point_indices.extend(idx_group)

        if not curb_point_indices:
            return np.zeros((0, 3), dtype=np.float32)

        return ground_points[np.array(curb_point_indices, dtype=int)]

    def detect_potholes(
        self,
        ground_points: np.ndarray,
        ground_plane: Optional[Tuple[float, float, float, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Identifies negative elevation depressions below expected ground surface.
        """
        x, y, z = ground_points[:, 0], ground_points[:, 1], ground_points[:, 2]

        if ground_plane is not None:
            a, b, c, d = ground_plane
            expected_z = -(a * x + b * y + d) / (c + 1e-6)
            depression = expected_z - z
        else:
            # Local moving plane / median reference
            tree = cKDTree(ground_points[:, :2])
            dists, idxs = tree.query(ground_points[:, :2], k=min(15, len(ground_points)))
            local_median_z = np.median(z[idxs], axis=1)
            depression = local_median_z - z

        pothole_candidate_mask = depression >= self.pothole_min_depth
        pothole_pts = ground_points[pothole_candidate_mask]

        if len(pothole_pts) < 4:
            return []

        # Cluster candidate points to identify discrete potholes
        clustering = DBSCAN(eps=0.8, min_samples=4).fit(pothole_pts[:, :2])
        labels = clustering.labels_
        potholes = []

        for lbl in set(labels) - {-1}:
            pts_cluster = pothole_pts[labels == lbl]
            dep_cluster = depression[pothole_candidate_mask][labels == lbl]
            
            center = np.mean(pts_cluster, axis=0)
            radius = float(np.max(np.linalg.norm(pts_cluster[:, :2] - center[:2], axis=1)))
            max_depth = float(np.max(dep_cluster))

            if radius < 2.5: # Plausible pothole radius
                potholes.append({
                    "center": center.tolist(),
                    "radius": radius,
                    "max_depth": max_depth,
                    "point_count": len(pts_cluster)
                })

        return potholes

    def detect_and_classify_clusters(
        self,
        non_ground_points: np.ndarray
    ) -> Tuple[List[ObstacleCluster], List[ObstacleCluster], List[ObstacleCluster]]:
        """
        Clusters non-ground obstacles and classifies them based on geometric PCA & bounding dimensions.
        """
        clustering = DBSCAN(eps=self.cluster_eps, min_samples=self.cluster_min_points).fit(non_ground_points)
        labels = clustering.labels_
        unique_labels = set(labels) - {-1}

        dynamic_obs = []
        infra_obs = []
        static_obs = []

        track_id = 1
        for lbl in unique_labels:
            pts = non_ground_points[labels == lbl]
            pos = np.mean(pts, axis=0)
            p_min = np.min(pts, axis=0)
            p_max = np.max(pts, axis=0)
            extents = p_max - p_min
            dx, dy, dz = extents[0], extents[1], extents[2]

            # Geometric PCA
            linearity, verticality = self._compute_pca_properties(pts)

            # Classification heuristics
            if dz >= 2.0 and max(dx, dy) <= 1.0 and verticality >= 0.7:
                category = "pole"
                infra_obs.append(ObstacleCluster(
                    cluster_id=track_id, category=category, position=pos, size=extents,
                    points=pts, linearity=linearity, verticality=verticality
                ))
            elif dz >= 2.5 and (dx >= 1.2 or dy >= 1.2):
                category = "tree"
                infra_obs.append(ObstacleCluster(
                    cluster_id=track_id, category=category, position=pos, size=extents,
                    points=pts, linearity=linearity, verticality=verticality
                ))
            elif (1.2 <= dz <= 2.2) and (0.3 <= dx <= 1.0) and (0.3 <= dy <= 1.0):
                category = "pedestrian"
                dynamic_obs.append(ObstacleCluster(
                    cluster_id=track_id, category=category, position=pos, size=extents,
                    points=pts, linearity=linearity, verticality=verticality
                ))
            elif (dx >= 2.2 or dy >= 2.2) and (dz >= 1.0):
                category = "vehicle"
                dynamic_obs.append(ObstacleCluster(
                    cluster_id=track_id, category=category, position=pos, size=extents,
                    points=pts, linearity=linearity, verticality=verticality
                ))
            else:
                category = "static_obstacle"
                static_obs.append(ObstacleCluster(
                    cluster_id=track_id, category=category, position=pos, size=extents,
                    points=pts, linearity=linearity, verticality=verticality
                ))

            track_id += 1

        return dynamic_obs, infra_obs, static_obs

    @staticmethod
    def _compute_pca_properties(points: np.ndarray) -> Tuple[float, float]:
        """Calculates linearity and vertical orientation via covariance eigen-decomposition."""
        if len(points) < 4:
            return 0.0, 0.0

        cov = np.cov(points.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Order descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = np.maximum(eigvals[idx], 1e-8)
        eigvecs = eigvecs[:, idx]

        l1, l2, l3 = eigvals[0], eigvals[1], eigvals[2]
        linearity = float((l1 - l2) / l1)

        # Alignment of principal direction with vertical axis [0, 0, 1]
        verticality = float(abs(eigvecs[2, 0]))
        return linearity, verticality
