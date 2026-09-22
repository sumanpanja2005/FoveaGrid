import numpy as np
from typing import Dict, Any

class LidarOdometry:
    """
    Ego-motion estimation interface.
    Can utilize ground-truth pose or ICP (Iterative Closest Point) scan matching.
    """

    def __init__(self, use_ground_truth: bool = True):
        self.use_ground_truth = use_ground_truth

    def estimate_relative_pose(
        self,
        prev_scan: Dict[str, Any],
        curr_scan: Dict[str, Any]
    ) -> np.ndarray:
        """
        Estimates relative SE(3) transformation matrix T_{prev -> curr}.
        """
        if self.use_ground_truth and "ego_pose" in prev_scan and "ego_pose" in curr_scan:
            T_prev = prev_scan["ego_pose"]
            T_curr = curr_scan["ego_pose"]
            # T_{prev -> curr} = inv(T_curr) @ T_prev
            R_curr_T = T_curr[:3, :3].T
            t_curr = T_curr[:3, 3]

            T_rel = np.eye(4, dtype=np.float32)
            T_rel[:3, :3] = R_curr_T @ T_prev[:3, :3]
            T_rel[:3, 3] = R_curr_T @ (T_prev[:3, 3] - t_curr)
            return T_rel

        # ICP Scan Matching Fallback
        pts_prev = prev_scan["points"]
        pts_curr = curr_scan["points"]
        return self._simple_icp(pts_prev, pts_curr)

    def _simple_icp(self, source: np.ndarray, target: np.ndarray, max_iters: int = 10) -> np.ndarray:
        """Lightweight SVD-based Point-to-Point ICP scan matcher fallback."""
        if len(source) < 50 or len(target) < 50:
            return np.eye(4, dtype=np.float32)

        from scipy.spatial import KDTree
        T = np.eye(4, dtype=np.float32)
        src = source[::5].copy() # Downsample for speed
        tree = KDTree(target[::5])

        for _ in range(max_iters):
            dists, indices = tree.query(src)
            matched_tgt = target[::5][indices]

            # Compute centroids
            centroid_src = np.mean(src, axis=0)
            centroid_tgt = np.mean(matched_tgt, axis=0)

            # Center points
            src_zero = src - centroid_src
            tgt_zero = matched_tgt - centroid_tgt

            # SVD decomposition
            H = src_zero.T @ tgt_zero
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            if np.linalg.det(R) < 0:
                Vt[2, :] *= -1
                R = Vt.T @ U.T

            t = centroid_tgt - R @ centroid_src

            # Update transformed source points
            src = (R @ src.T).T + t

            T_step = np.eye(4, dtype=np.float32)
            T_step[:3, :3] = R
            T_step[:3, 3] = t
            T = T_step @ T

        return T
