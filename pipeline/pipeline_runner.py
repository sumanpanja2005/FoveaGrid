import time
import numpy as np
from typing import Union, Dict, Any, Optional, Tuple

from .ingestion import LidarDataIngestor, PointCloudData
from .preprocessing import (
    voxel_downsample,
    statistical_outlier_removal,
    segment_ground_ransac,
    segment_ground_pmf
)
from .feature_extraction import RoadFeatureExtractor, DetectedFeatures
from .bev_projection import BEVProjector, BEVGrid
from .dem_elevation import DEMGenerator, DEMGrid
from .renderer_25d import ElevationRenderer25D

class LidarPerceptionPipeline:
    """
    End-to-End Autonomous Vehicle Roadside 3D LiDAR Perception Pipeline:
    1. Ingestion (.pcd, .las, .npy, PointCloud2)
    2. Voxel Downsampling & Outlier Filtering
    3. Ground Plane Segmentation (RANSAC / PMF)
    4. Multi-Class Feature Extraction (Curbs, Potholes, Vehicles, Pedestrians, Infrastructure)
    5. 2D Bird's Eye View (BEV) Multi-Channel Rasterization
    6. 2.5D Digital Elevation Model (DEM) Surface Generation
    7. False-Color Isometric 3D Visualization
    """

    def __init__(
        self,
        voxel_size: float = 0.10,
        sor_k: int = 20,
        sor_std: float = 2.0,
        ground_method: str = "ransac",
        ransac_distance: float = 0.15,
        ransac_normal_tol: float = 15.0,
        bev_bounds: Tuple[float, float, float, float] = (-40.0, 40.0, -40.0, 40.0),
        bev_resolution: float = 0.20,
        dem_resolution: float = 0.25,
        inpaint_dem: bool = True
    ):
        self.voxel_size = voxel_size
        self.sor_k = sor_k
        self.sor_std = sor_std
        self.ground_method = ground_method.lower()
        self.ransac_distance = ransac_distance
        self.ransac_normal_tol = ransac_normal_tol
        self.bev_bounds = bev_bounds
        self.bev_resolution = bev_resolution
        self.dem_resolution = dem_resolution

        self.feature_extractor = RoadFeatureExtractor()
        self.bev_projector = BEVProjector(bounds=bev_bounds, resolution=bev_resolution)
        self.dem_generator = DEMGenerator(bounds=bev_bounds, resolution=dem_resolution, inpaint_holes=inpaint_dem)
        self.renderer = ElevationRenderer25D()

    def process(self, input_data: Union[str, Dict[str, Any], np.ndarray, PointCloudData]) -> Dict[str, Any]:
        """
        Executes end-to-end processing pipeline on a single LiDAR scan.
        """
        timings = {}
        t_total_start = time.perf_counter()

        # 1. Ingestion
        t0 = time.perf_counter()
        if isinstance(input_data, PointCloudData):
            pcd = input_data
        else:
            pcd = LidarDataIngestor.load(input_data)
        timings["ingestion_ms"] = (time.perf_counter() - t0) * 1000.0
        raw_count = len(pcd.points)

        # 2. Voxel Downsampling
        t0 = time.perf_counter()
        if self.voxel_size > 0.0:
            ds_points, ds_intensity, _ = voxel_downsample(pcd.points, voxel_size=self.voxel_size, intensity=pcd.intensity)
        else:
            ds_points, ds_intensity = pcd.points, pcd.intensity
        timings["downsampling_ms"] = (time.perf_counter() - t0) * 1000.0

        # 3. Statistical Outlier Removal
        t0 = time.perf_counter()
        if self.sor_k > 0:
            filtered_points, filtered_intensity, _ = statistical_outlier_removal(
                ds_points, k_neighbors=self.sor_k, std_ratio=self.sor_std, intensity=ds_intensity
            )
        else:
            filtered_points, filtered_intensity = ds_points, ds_intensity
        timings["outlier_removal_ms"] = (time.perf_counter() - t0) * 1000.0

        # 4. Ground Plane Segmentation
        t0 = time.perf_counter()
        ground_plane = None
        if self.ground_method == "pmf":
            ground_mask, obstacle_mask = segment_ground_pmf(filtered_points)
        else:
            ground_mask, obstacle_mask, ground_plane = segment_ground_ransac(
                filtered_points,
                distance_threshold=self.ransac_distance,
                normal_tolerance_deg=self.ransac_normal_tol
            )
        timings["ground_segmentation_ms"] = (time.perf_counter() - t0) * 1000.0

        ground_pts = filtered_points[ground_mask]
        obstacle_pts = filtered_points[obstacle_mask]

        # 5. Multi-Class Feature Extraction
        t0 = time.perf_counter()
        features = self.feature_extractor.extract_features(
            ground_points=ground_pts,
            non_ground_points=obstacle_pts,
            ground_plane=ground_plane
        )
        timings["feature_extraction_ms"] = (time.perf_counter() - t0) * 1000.0

        # 6. 2D Bird's Eye View (BEV) Projection
        t0 = time.perf_counter()
        bev_grid = self.bev_projector.project(
            points=filtered_points,
            ground_mask=ground_mask,
            intensities=filtered_intensity
        )
        timings["bev_projection_ms"] = (time.perf_counter() - t0) * 1000.0

        # 7. 2.5D Digital Elevation Model (DEM)
        t0 = time.perf_counter()
        dem_grid = self.dem_generator.generate(
            points=filtered_points,
            ground_mask=ground_mask
        )
        timings["dem_generation_ms"] = (time.perf_counter() - t0) * 1000.0

        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        timings["total_pipeline_ms"] = total_ms

        return {
            "raw_points_count": raw_count,
            "processed_points_count": len(filtered_points),
            "ground_points_count": int(np.sum(ground_mask)),
            "obstacle_points_count": int(np.sum(obstacle_mask)),
            "ground_plane": ground_plane,
            "filtered_points": filtered_points,
            "ground_mask": ground_mask,
            "features": features,
            "bev": bev_grid,
            "dem": dem_grid,
            "timings": timings,
            "fps": round(1000.0 / max(0.1, total_ms), 1)
        }

    def render(
        self,
        result: Dict[str, Any],
        output_image_path: Optional[str] = None,
        dashboard: bool = True,
        show_interactive: bool = False
    ):
        """
        Renders the pipeline output to visual plots.
        """
        dem: DEMGrid = result["dem"]
        bev: BEVGrid = result["bev"]
        features: DetectedFeatures = result["features"]

        if dashboard:
            self.renderer.render_comprehensive_dashboard(
                dem=dem, bev=bev, features=features,
                save_path=output_image_path, show_interactive=show_interactive
            )
        else:
            self.renderer.render_isometric_dem(
                dem=dem, features=features,
                save_path=output_image_path, show_interactive=show_interactive
            )
