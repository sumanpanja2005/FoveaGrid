import unittest
import numpy as np
from simulation.environment import SyntheticEnvironment
from simulation.lidar_simulator import LidarSimulator

class TestLidarSimulation(unittest.TestCase):
    def test_lidar_simulation(self):
        env = SyntheticEnvironment(scene_id=1, seed=42)
        sim = LidarSimulator(beams=64)
        scan = sim.simulate_scan(env)

        self.assertIn("points", scan)
        self.assertIn("labels", scan)
        self.assertIn("ring", scan)

        pts = scan["points"]
        self.assertGreaterThan(len(pts), 0) if hasattr(self, 'assertGreaterThan') else self.assertTrue(len(pts) > 0)
        self.assertEqual(pts.shape[1], 3)

        rings = scan["ring"]
        self.assertTrue(np.min(rings) >= 0)
        self.assertTrue(np.max(rings) < 64)

    def test_noise_model(self):
        env = SyntheticEnvironment(scene_id=1, seed=42)
        sim = LidarSimulator(beams=64)
        scan = sim.simulate_scan(env)

        pts = scan["points"]
        ranges = np.linalg.norm(pts, axis=1)
        self.assertTrue(np.all(ranges >= 0.5))
        self.assertTrue(np.all(ranges <= 100.0))

if __name__ == "__main__":
    unittest.main()
