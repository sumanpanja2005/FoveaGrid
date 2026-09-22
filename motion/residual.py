import numpy as np

class MotionResidualAnalyzer:
    """
    Computes point-level motion residual:
    motion_residual = observed_motion - ego_motion
    Combines learned semantic probability with temporal motion residual to identify dynamic objects.
    """

    def __init__(self, residual_threshold: float = 0.35):
        self.residual_threshold = residual_threshold

    def compute_residuals(
        self,
        curr_points: np.ndarray,
        prev_points: np.ndarray,
        T_prev_to_curr: np.ndarray
    ) -> np.ndarray:
        """
        Computes distance residual from nearest neighbor in motion-compensated previous scan.
        """
        if len(curr_points) == 0 or len(prev_points) == 0:
            return np.zeros(len(curr_points), dtype=np.float32)

        # 1. Transform previous points into current frame using Ego-motion T_{prev -> curr}
        R = T_prev_to_curr[:3, :3]
        t = T_prev_to_curr[:3, 3]
        prev_warped = (R @ prev_points.T).T + t

        # 2. Nearest neighbor distance search in warped previous scan
        from scipy.spatial import KDTree
        tree = KDTree(prev_warped)
        dists, _ = tree.query(curr_points)

        return dists.astype(np.float32)

    def compute_dynamic_probability(
        self,
        semantic_probs: np.ndarray, # Nxnum_classes
        motion_residuals: np.ndarray
    ) -> np.ndarray:
        """
        Fuses learned semantic probability (class 3: dynamic) with temporal motion residual.
        """
        sem_dynamic_prob = semantic_probs[:, 3] if semantic_probs.shape[1] > 3 else np.zeros(len(semantic_probs))
        
        # Sigmoid scaling on motion residual
        residual_prob = 1.0 / (1.0 + np.exp(-10.0 * (motion_residuals - self.residual_threshold)))

        # Weighted combination
        fused_dynamic_prob = 0.5 * sem_dynamic_prob + 0.5 * residual_prob
        return fused_dynamic_prob.astype(np.float32)
