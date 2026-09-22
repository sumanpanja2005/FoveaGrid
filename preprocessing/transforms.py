import numpy as np

def transform_point_cloud(points: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Applies 4x4 SE(3) rigid transformation matrix to Nx3 point cloud.
    """
    if len(points) == 0:
        return points
    R = transform_matrix[:3, :3]
    t = transform_matrix[:3, 3]
    return (R @ points.T).T + t

def compose_se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Creates 4x4 SE(3) matrix from 3x3 rotation matrix R and 3-vector translation t."""
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def invert_se3(T: np.ndarray) -> np.ndarray:
    """Inverts 4x4 SE(3) matrix."""
    T_inv = np.eye(4, dtype=np.float32)
    R_T = T[:3, :3].T
    T_inv[:3, :3] = R_T
    T_inv[:3, 3] = -R_T @ T[:3, 3]
    return T_inv
