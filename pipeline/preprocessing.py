import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import grey_erosion, grey_dilation
from typing import Tuple, Optional

def voxel_downsample(
    points: np.ndarray,
    voxel_size: float = 0.1,
    intensity: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Downsamples point cloud using uniform spatial voxel hashing with centroid aggregation.
    
    Returns:
    - downsampled_points: (M, 3)
    - downsampled_intensity: (M,) or None
    - point_to_voxel: (N,) index mapping from original to downsampled
    """
    if len(points) == 0:
        return points, intensity, np.zeros(0, dtype=np.int64)

    coords = np.floor(points / voxel_size).astype(np.int64)
    unique_coords, inverse_indices = np.unique(coords, axis=0, return_inverse=True)
    M = len(unique_coords)

    # Fast centroid calculation using bincount / scatter
    counts = np.bincount(inverse_indices, minlength=M).astype(np.float32)[:, None]
    
    sum_x = np.bincount(inverse_indices, weights=points[:, 0], minlength=M)
    sum_y = np.bincount(inverse_indices, weights=points[:, 1], minlength=M)
    sum_z = np.bincount(inverse_indices, weights=points[:, 2], minlength=M)
    
    downsampled_points = np.column_stack([sum_x, sum_y, sum_z]) / counts

    downsampled_intensity = None
    if intensity is not None and len(intensity) == len(points):
        sum_i = np.bincount(inverse_indices, weights=intensity, minlength=M)
        downsampled_intensity = (sum_i / counts[:, 0]).astype(np.float32)

    return downsampled_points.astype(np.float32), downsampled_intensity, inverse_indices

def statistical_outlier_removal(
    points: np.ndarray,
    k_neighbors: int = 20,
    std_ratio: float = 2.0,
    intensity: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Filters isolated noise points based on mean k-nearest neighbor distance distribution.
    
    Returns:
    - inlier_points: (K, 3)
    - inlier_intensity: (K,) or None
    - inlier_mask: (N,) boolean mask
    """
    N = len(points)
    if N <= k_neighbors:
        mask = np.ones(N, dtype=bool)
        return points, intensity, mask

    tree = cKDTree(points)
    # Query k+1 neighbors (0th is the point itself)
    dists, _ = tree.query(points, k=k_neighbors + 1, workers=-1)
    mean_dists = np.mean(dists[:, 1:], axis=1)

    mu = np.mean(mean_dists)
    sigma = np.std(mean_dists)
    threshold = mu + std_ratio * sigma

    inlier_mask = mean_dists <= threshold
    inlier_points = points[inlier_mask]
    inlier_intensity = intensity[inlier_mask] if intensity is not None else None

    return inlier_points, inlier_intensity, inlier_mask

def segment_ground_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.15,
    max_iterations: int = 150,
    normal_tolerance_deg: float = 15.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float, float, float]]:
    """
    Segments ground plane using normal-constrained RANSAC plane fitting:
    ax + by + cz + d = 0, where normal vector [a, b, c] must be approximately vertical.
    
    Returns:
    - ground_mask: (N,) boolean mask
    - obstacle_mask: (N,) boolean mask
    - plane_model: (a, b, c, d) normalized coefficients
    """
    N = len(points)
    if N < 3:
        mask = np.zeros(N, dtype=bool)
        return mask, ~mask, (0.0, 0.0, 1.0, 0.0)

    rng = np.random.default_rng(seed)
    cos_tol = np.cos(np.radians(normal_tolerance_deg))

    best_inliers = np.zeros(N, dtype=bool)
    best_inlier_count = 0
    best_plane = (0.0, 0.0, 1.0, 0.0)

    # Initial height-based candidate pruning (ground is typically near or below sensor z=0)
    low_idx = np.where(points[:, 2] < np.median(points[:, 2]))[0]
    sample_pool = low_idx if len(low_idx) >= 10 else np.arange(N)

    for _ in range(max_iterations):
        sample_indices = rng.choice(sample_pool, size=3, replace=False)
        p1, p2, p3 = points[sample_indices]

        # Calculate plane normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-6:
            continue

        normal = normal / norm_len

        # Enforce vertical normal alignment (upward Z)
        if normal[2] < 0:
            normal = -normal

        if normal[2] < cos_tol:
            continue  # Normal is tilted too far away from vertical

        d = -np.dot(normal, p1)

        # Distance from all points to plane: |ax + by + cz + d|
        distances = np.abs(np.dot(points, normal) + d)
        inlier_mask = distances <= distance_threshold
        inlier_count = np.sum(inlier_mask)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_inliers = inlier_mask
            best_plane = (float(normal[0]), float(normal[1]), float(normal[2]), float(d))

    # Refine plane model using all inliers
    if best_inlier_count >= 3:
        inlier_pts = points[best_inliers]
        centroid = np.mean(inlier_pts, axis=0)
        centered = inlier_pts - centroid
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        refined_normal = eigvecs[:, 0]  # Smallest eigenvalue corresponds to plane normal
        if refined_normal[2] < 0:
            refined_normal = -refined_normal
        if refined_normal[2] >= cos_tol:
            refined_d = -np.dot(refined_normal, centroid)
            best_plane = (float(refined_normal[0]), float(refined_normal[1]), float(refined_normal[2]), float(refined_d))

    final_normal = np.array(best_plane[:3], dtype=np.float32)
    final_d = best_plane[3]
    signed_dist = np.dot(points, final_normal) + final_d

    # Ground includes flat road and negative depressions/potholes down to 0.35m
    ground_mask = (signed_dist >= -0.35) & (signed_dist <= distance_threshold)
    obstacle_mask = signed_dist > distance_threshold

    return ground_mask, obstacle_mask, best_plane

def segment_ground_pmf(
    points: np.ndarray,
    cell_size: float = 0.5,
    max_window_size: int = 15,
    slope_threshold: float = 0.15,
    initial_elevation_threshold: float = 0.15,
    max_elevation_threshold: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Progressive Morphological Filtering (PMF) for ground segmentation across sloped or uneven terrain.
    Uses progressive morphological opening (erosion + dilation) with expanding structural element kernels.
    
    Returns:
    - ground_mask: (N,) boolean
    - obstacle_mask: (N,) boolean
    """
    N = len(points)
    if N == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    grid_w = max(1, int(np.ceil((x_max - x_min) / cell_size)))
    grid_h = max(1, int(np.ceil((y_max - y_min) / cell_size)))

    ix = np.clip(np.floor((x - x_min) / cell_size).astype(int), 0, grid_w - 1)
    iy = np.clip(np.floor((y - y_min) / cell_size).astype(int), 0, grid_h - 1)

    # Initialize raster grid with min height
    grid_z_min = np.full((grid_h, grid_w), np.nan, dtype=np.float32)
    # Flattened indices for 2D aggregation
    flat_idx = iy * grid_w + ix
    order = np.argsort(flat_idx)
    sorted_flat = flat_idx[order]
    sorted_z = z[order]

    unique_bins, split_idx = np.unique(sorted_flat, return_index=True)
    bin_mins = np.minimum.reduceat(sorted_z, split_idx)

    u_iy = unique_bins // grid_w
    u_ix = unique_bins % grid_w
    grid_z_min[u_iy, u_ix] = bin_mins

    # Inpaint empty raster cells via local nearest interpolation
    nan_mask = np.isnan(grid_z_min)
    if np.any(nan_mask):
        valid_coords = np.column_stack(np.where(~nan_mask))
        if len(valid_coords) > 0:
            nan_coords = np.column_stack(np.where(nan_mask))
            tree = cKDTree(valid_coords)
            _, nearest_idx = tree.query(nan_coords)
            grid_z_min[nan_mask] = grid_z_min[valid_coords[nearest_idx, 0], valid_coords[nearest_idx, 1]]
        else:
            grid_z_min[nan_mask] = 0.0

    # Iterative Morphological Filtering
    current_surface = grid_z_min.copy()
    window_sizes = [2 * k + 1 for k in range(1, (max_window_size // 2) + 1)]

    for k, w in enumerate(window_sizes):
        # Morphological opening = erosion followed by dilation
        eroded = grey_erosion(current_surface, size=(w, w))
        opened = grey_dilation(eroded, size=(w, w))
        
        # Adaptive elevation threshold
        elev_thresh = min(
            max_elevation_threshold,
            slope_threshold * (w * cell_size) + initial_elevation_threshold
        )

        diff = current_surface - opened
        current_surface = np.where(diff > elev_thresh, opened, current_surface)

    # Compare point heights to final filtered ground surface
    surface_z_at_pts = current_surface[iy, ix]
    elevation_diff = z - surface_z_at_pts
    ground_mask = (elevation_diff >= -0.3) & (elevation_diff <= initial_elevation_threshold)

    return ground_mask, ~ground_mask
