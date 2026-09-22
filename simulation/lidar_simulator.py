import numpy as np
from typing import Dict, Any, Tuple
from .environment import SyntheticEnvironment
from .noise_model import LidarNoiseModel

class LidarSimulator:
    """
    Simulates a 64-beam rotating automotive LiDAR.
    Vertical FOV: -25° to +15°
    Horizontal FOV: 360°
    Outputs point clouds with fields: [x, y, z, intensity, timestamp, ring, semantic_label].
    """

    def __init__(
        self,
        beams: int = 64,
        fov_up: float = 15.0,
        fov_down: float = -25.0,
        max_range: float = 100.0,
        h_res_deg: float = 0.2,
        scan_rate: int = 10,
        noise_model: LidarNoiseModel = None
    ):
        self.beams = beams
        self.max_range = max_range
        self.scan_rate = scan_rate
        self.noise_model = noise_model or LidarNoiseModel()

        # Vertical elevation angles per beam (ring)
        self.ring_elevations = np.radians(np.linspace(fov_down, fov_up, beams))
        # Horizontal azimuth angles
        self.azimuths = np.radians(np.arange(0.0, 360.0, h_res_deg))

        # Precompute unit ray directions in LiDAR local frame
        # Shape: (beams, num_azimuths, 3)
        grid_el, grid_az = np.meshgrid(self.ring_elevations, self.azimuths, indexing='ij')
        cos_el = np.cos(grid_el)
        
        rx = cos_el * np.cos(grid_az)
        ry = cos_el * np.sin(grid_az)
        rz = np.sin(grid_el)

        self.ray_dirs_local = np.stack([rx, ry, rz], axis=-1) # (64, N_az, 3)
        self.rings_grid = np.repeat(np.arange(beams)[:, None], len(self.azimuths), axis=1)

    def simulate_scan(self, env: SyntheticEnvironment, timestamp: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Casts 64-beam rays into synthetic environment, intersects terrain & objects, and returns point cloud.
        """
        # Transform ray directions into World Frame using Ego Pose
        ego_rot = env.ego_pose[:3, :3]
        ego_trans = env.ego_pose[:3, 3]

        flat_dirs_local = self.ray_dirs_local.reshape(-1, 3)
        flat_rings = self.rings_grid.reshape(-1)
        flat_dirs_world = (ego_rot @ flat_dirs_local.T).T

        N_rays = len(flat_dirs_world)
        ray_origins = np.tile(ego_trans, (N_rays, 1))

        # Initialize hit distances with max range
        hit_distances = np.full(N_rays, self.max_range, dtype=np.float32)
        hit_labels = np.zeros(N_rays, dtype=np.int32) # 0 = unknown
        hit_intensities = np.full(N_rays, 0.2, dtype=np.float32)

        # 1. Terrain Surface Ray-Casting
        # Intersect rays with ground plane / terrain function
        # Using fast line-plane step approximation for ground terrain
        dz = flat_dirs_world[:, 2]
        downward = dz < -1e-4
        if np.any(downward):
            # Distance to z=0 surface approximation
            t_ground = (0.0 - ray_origins[downward, 2]) / dz[downward]
            valid_g = (t_ground > 0.5) & (t_ground < self.max_range)
            
            idx_g = np.where(downward)[0][valid_g]
            t_g_valid = t_ground[valid_g]
            
            # Compute intersection coordinates in world frame
            pts_world_g = ray_origins[idx_g] + t_g_valid[:, None] * flat_dirs_world[idx_g]
            z_terr, labels_terr = env.terrain.get_ground_elevation(pts_world_g[:, 0], pts_world_g[:, 1])

            # Refine intersection distance
            t_refined = (z_terr - ray_origins[idx_g, 2]) / dz[idx_g]
            valid_ref = (t_refined > 0.5) & (t_refined < hit_distances[idx_g])
            
            hit_distances[idx_g[valid_ref]] = t_refined[valid_ref]
            hit_labels[idx_g[valid_ref]] = labels_terr[valid_ref]
            hit_intensities[idx_g[valid_ref]] = 0.4 # Terrain reflectivity

        # 2. Intersect Physical Objects (Static, Dynamic, Overhead)
        for obj in env.objects:
            obj_dists = obj.ray_intersection(ray_origins, flat_dirs_world)
            closer = (obj_dists > 0.5) & (obj_dists < hit_distances)
            if np.any(closer):
                hit_distances[closer] = obj_dists[closer]
                hit_labels[closer] = obj.class_id
                hit_intensities[closer] = 0.8 if obj.category == "dynamic" else 0.6

        # Filter valid returns (range < max_range)
        valid_mask = hit_distances < (self.max_range - 0.5)
        
        valid_dists = hit_distances[valid_mask]
        valid_dirs_local = flat_dirs_local[valid_mask]
        valid_labels = hit_labels[valid_mask]
        valid_intensities = hit_intensities[valid_mask]
        valid_rings = flat_rings[valid_mask]

        # Calculate Cartesian point positions in LiDAR local frame
        points_local = valid_dirs_local * valid_dists[:, None]

        # Apply Realistic Sensor Noise & Dropout (preserving exact beam ring assignment)
        noisy_points, noisy_intensities, final_labels, rings_output = self.noise_model.apply_noise(
            points_local, valid_intensities, valid_labels, rings=valid_rings
        )
        
        # Calculate timestamps (spinning rotation sweep)
        azimuths_valid = np.arctan2(noisy_points[:, 1], noisy_points[:, 0]) % (2 * np.pi)
        timestamps = timestamp + (azimuths_valid / (2 * np.pi)) * (1.0 / self.scan_rate)

        return {
            "points": noisy_points.astype(np.float32), # Nx3 LiDAR frame
            "intensity": noisy_intensities.astype(np.float32),
            "timestamp": timestamps.astype(np.float32),
            "ring": rings_output.astype(np.int32),
            "labels": final_labels.astype(np.int32),
            "ego_pose": env.ego_pose.copy(),
            "gt_boxes": env.get_ground_truth_boxes()
        }
