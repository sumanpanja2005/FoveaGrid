import os
import unittest
import numpy as np

from pipeline.ingestion import LidarDataIngestor, PointCloudData, SyntheticStreamGenerator
from pipeline.pipeline_runner import LidarPerceptionPipeline
from pipeline.payload import Unified25DPayloadPacker

class TestPayloadAndStream(unittest.TestCase):

    def setUp(self):
        self.temp_dir = "data/test_formats"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.generator = SyntheticStreamGenerator(seed=42)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            for f in os.listdir(self.temp_dir):
                os.remove(os.path.join(self.temp_dir, f))
            os.rmdir(self.temp_dir)

    def test_synthetic_stream_generation(self):
        frame = self.generator.next_frame(dt=0.1)
        self.assertGreater(len(frame.points), 1000)
        self.assertEqual(len(frame.points), len(frame.intensity))
        self.assertIn("synthetic_stream", frame.metadata.get("format", ""))
        self.assertAlmostEqual(frame.metadata.get("timestamp", 0.0), 0.1, delta=0.01)

    def test_ply_save_and_load(self):
        ply_path = os.path.join(self.temp_dir, "test_scan.ply")
        frame = self.generator.next_frame(dt=0.1)
        # Save as binary PLY
        LidarDataIngestor.save_ply(frame, ply_path, mode="binary")
        self.assertTrue(os.path.exists(ply_path))

        # Auto-detect and load
        loaded = LidarDataIngestor.load(ply_path)
        self.assertEqual(len(loaded.points), len(frame.points))
        self.assertEqual(loaded.metadata.get("format"), "ply")
        np.testing.assert_allclose(loaded.points[:50], frame.points[:50], atol=1e-3)

    def test_bin_save_and_load(self):
        bin_path = os.path.join(self.temp_dir, "test_scan.bin")
        frame = self.generator.next_frame(dt=0.1)
        LidarDataIngestor.save_bin(frame, bin_path)
        self.assertTrue(os.path.exists(bin_path))

        loaded = LidarDataIngestor.load(bin_path)
        self.assertEqual(len(loaded.points), len(frame.points))
        self.assertEqual(loaded.metadata.get("format"), "bin")
        np.testing.assert_allclose(loaded.points[:50], frame.points[:50], atol=1e-3)

    def test_unified_payload_packing(self):
        pipeline = LidarPerceptionPipeline(voxel_size=0.2, dem_resolution=0.5, bev_resolution=0.5)
        frame = self.generator.next_frame(dt=0.1)
        result = pipeline.process(frame)

        payload = Unified25DPayloadPacker.pack(
            dem=result["dem"],
            features=result["features"],
            bev=result["bev"],
            timings=result["timings"],
            raw_count=result["raw_points_count"],
            processed_count=result["processed_points_count"],
            ground_count=result["ground_points_count"],
            source_name="Test Generator",
            source_format="Synthetic"
        )

        self.assertEqual(payload["type"], "unified_25d_frame")
        self.assertIn("dem", payload)
        self.assertIn("curbs", payload)
        self.assertIn("potholes", payload)
        self.assertIn("obstacles", payload)
        self.assertIn("telemetry", payload)

        # Check DEM array structure
        dem_info = payload["dem"]
        self.assertGreater(dem_info["width"], 0)
        self.assertGreater(dem_info["height"], 0)
        self.assertEqual(len(dem_info["elevation"]), dem_info["width"] * dem_info["height"])
        self.assertLess(dem_info["min_z"], dem_info["max_z"])

        # Check telemetry
        telemetry = payload["telemetry"]
        self.assertEqual(telemetry["source"], "Test Generator")
        self.assertGreater(telemetry["fps"], 0.0)

if __name__ == "__main__":
    unittest.main()
