import os
import unittest
import numpy as np

from pipeline.ingestion import LidarDataIngestor, PointCloudData
from pipeline.preprocessing import (
    voxel_downsample,
    statistical_outlier_removal,
    segment_ground_ransac,
    segment_ground_pmf
)
from pipeline.feature_extraction import RoadFeatureExtractor
from pipeline.bev_projection import BEVProjector
from pipeline.dem_elevation import DEMGenerator
from pipeline.pipeline_runner import LidarPerceptionPipeline

class TestLidarPipeline(unittest.TestCase):

    def setUp(self):
        np.random.seed(42)
        # Create a synthetic road scene
        # 1. Ground points: flat plane at z = 0 with slight noise
        N_ground = 2000
        gx = np.random.uniform(-20.0, 20.0, N_ground)
        gy = np.random.uniform(-10.0, 10.0, N_ground)
        gz = np.random.normal(0.0, 0.02, N_ground)

        # 2. Curb: step of 15cm along y = 5.0
        curb_mask = (gy >= 4.8) & (gy <= 5.2)
        gz[curb_mask] += 0.15

        # 3. Pothole: circular depression at (x=5.0, y=0.0) with depth 0.18m
        dist_ph = np.hypot(gx - 5.0, gy - 0.0)
        in_ph = dist_ph < 0.8
        gz[in_ph] -= 0.18

        ground_pts = np.column_stack([gx, gy, gz])

        # 4. Pole: tall vertical cylinder at (-5, 6, 0..3m)
        N_pole = 100
        px = np.full(N_pole, -5.0) + np.random.normal(0, 0.05, N_pole)
        py = np.full(N_pole, 6.0) + np.random.normal(0, 0.05, N_pole)
        pz = np.linspace(0.2, 3.5, N_pole)
        pole_pts = np.column_stack([px, py, pz])

        # 5. Vehicle: rectangular box at (10, -2, 0.5..2.0m)
        N_veh = 300
        vx = np.random.uniform(8.0, 12.0, N_veh)
        vy = np.random.uniform(-3.0, -1.0, N_veh)
        vz = np.random.uniform(0.3, 1.8, N_veh)
        veh_pts = np.column_stack([vx, vy, vz])

        self.all_points = np.vstack([ground_pts, pole_pts, veh_pts]).astype(np.float32)
        self.intensities = np.random.uniform(0.1, 0.9, len(self.all_points)).astype(np.float32)

    def test_pcd_save_and_load(self):
        temp_pcd = "data/test_temp.pcd"
        pcd_in = PointCloudData(points=self.all_points[:100], intensity=self.intensities[:100])
        LidarDataIngestor.save_pcd(pcd_in, temp_pcd, mode="ascii")
        self.assertTrue(os.path.exists(temp_pcd))

        pcd_out = LidarDataIngestor.load_pcd(temp_pcd)
        self.assertEqual(len(pcd_out.points), 100)
        np.testing.assert_allclose(pcd_in.points, pcd_out.points, atol=1e-3)

        if os.path.exists(temp_pcd):
            os.remove(temp_pcd)

    def test_voxel_downsampling(self):
        ds_pts, ds_inten, mapping = voxel_downsample(self.all_points, voxel_size=0.5, intensity=self.intensities)
        self.assertLess(len(ds_pts), len(self.all_points))
        self.assertEqual(len(ds_pts), len(ds_inten))
        self.assertEqual(len(mapping), len(self.all_points))

    def test_statistical_outlier_removal(self):
        # Add 3 extreme outlier points
        outliers = np.array([[100.0, 100.0, 50.0], [-100.0, -100.0, 50.0], [0.0, 100.0, 50.0]])
        noisy_pts = np.vstack([self.all_points, outliers])
        filtered_pts, _, mask = statistical_outlier_removal(noisy_pts, k_neighbors=10, std_ratio=1.5)
        # Verify outliers were removed
        self.assertFalse(mask[-1])
        self.assertFalse(mask[-2])
        self.assertFalse(mask[-3])

    def test_ground_segmentation_ransac(self):
        g_mask, obs_mask, plane = segment_ground_ransac(self.all_points, distance_threshold=0.15)
        self.assertTrue(np.sum(g_mask) > 1500, "Ground plane should capture majority of ground points")
        self.assertTrue(np.sum(obs_mask) >= 300, "Obstacle points should capture pole and vehicle")
        # Ground plane normal should be vertical
        self.assertGreater(abs(plane[2]), 0.95)

    def test_ground_segmentation_pmf(self):
        g_mask, obs_mask = segment_ground_pmf(self.all_points, cell_size=0.5)
        self.assertTrue(np.sum(g_mask) > 1000)
        self.assertTrue(np.sum(obs_mask) > 100)

    def test_feature_extraction(self):
        extractor = RoadFeatureExtractor()
        g_mask, obs_mask, plane = segment_ground_ransac(self.all_points, distance_threshold=0.15)
        features = extractor.extract_features(
            ground_points=self.all_points[g_mask],
            non_ground_points=self.all_points[obs_mask],
            ground_plane=plane
        )

        # Check pothole detection
        self.assertGreaterEqual(len(features.potholes), 1)
        ph = features.potholes[0]
        self.assertAlmostEqual(ph["center"][0], 5.0, delta=1.5)
        self.assertAlmostEqual(ph["center"][1], 0.0, delta=1.5)

        # Check curb points
        self.assertGreater(len(features.curb_points), 0)

        # Check obstacle classification (pole & vehicle)
        all_clusters = features.dynamic_obstacles + features.roadside_infrastructure + features.static_obstacles
        categories = [c.category for c in all_clusters]
        self.assertTrue("pole" in categories or "vehicle" in categories)

    def test_bev_projection(self):
        projector = BEVProjector(bounds=(-20.0, 20.0, -10.0, 10.0), resolution=0.5)
        g_mask, _, _ = segment_ground_ransac(self.all_points)
        bev = projector.project(self.all_points, ground_mask=g_mask, intensities=self.intensities)

        self.assertEqual(bev.tensor.shape[2], 6)
        self.assertTrue(np.all(bev.z_max >= bev.z_min - 1e-4))
        self.assertTrue(np.any(bev.density > 0))

    def test_dem_elevation_grid(self):
        generator = DEMGenerator(bounds=(-20.0, 20.0, -10.0, 10.0), resolution=0.5, inpaint_holes=True)
        g_mask, _, _ = segment_ground_ransac(self.all_points)
        dem = generator.generate(self.all_points, ground_mask=g_mask)

        # Inpainting should eliminate all NaNs
        self.assertFalse(np.any(np.isnan(dem.elevation)))
        self.assertEqual(dem.elevation.shape, (40, 80))
        # Obstacle height should be positive where vehicle/pole is
        self.assertGreater(np.max(dem.obstacle_height), 1.0)

    def test_end_to_end_pipeline(self):
        pipeline = LidarPerceptionPipeline(
            voxel_size=0.2,
            bev_bounds=(-20.0, 20.0, -10.0, 10.0),
            bev_resolution=0.5,
            dem_resolution=0.5
        )
        result = pipeline.process(self.all_points)
        self.assertIn("raw_points_count", result)
        self.assertIn("features", result)
        self.assertIn("bev", result)
        self.assertIn("dem", result)
        self.assertGreater(result["fps"], 0.0)

if __name__ == "__main__":
    unittest.main()
