import unittest
import numpy as np
from mapping.fovea_grid import FoveaGrid
from mapping.micro_fovea import MicroFoveaAllocator

class TestFovea(unittest.TestCase):
    def test_adaptive_resolution_bands(self):
        grid = FoveaGrid()
        ego_pos = np.array([0.0, 0.0, 0.0])
        ego_vel = np.array([0.0, 0.0, 0.0])

        res_near = grid.get_desired_resolution(2.0, 2.0, ego_pos, ego_vel)
        self.assertEqual(res_near, 0.05) # 5cm inside 10m

        res_mid = grid.get_desired_resolution(30.0, 0.0, ego_pos, ego_vel)
        self.assertEqual(res_mid, 0.20) # 20cm at 30m

        res_far = grid.get_desired_resolution(80.0, 0.0, ego_pos, ego_vel)
        self.assertEqual(res_far, 0.50) # 50cm at 80m

    def test_speed_steering(self):
        grid = FoveaGrid()
        ego_pos = np.array([0.0, 0.0, 0.0])
        ego_vel_fast = np.array([20.0, 0.0, 0.0]) # 20 m/s forward speed

        res_ahead = grid.get_desired_resolution(20.0, 0.0, ego_pos, ego_vel_fast)
        self.assertEqual(res_ahead, 0.05) # Detailed 5cm fovea shifted forward

    def test_micro_fovea_allocation(self):
        allocator = MicroFoveaAllocator()
        ego_pos = np.array([0.0, 0.0, 0.0])
        targets = [{"position": np.array([35.0, 2.0, 0.0])}] # Target at 35m

        allocator.update_micro_foveas(targets, ego_pos)
        self.assertEqual(len(allocator.active_micro_foveas), 1)
        self.assertTrue(allocator.is_in_micro_fovea(35.0, 2.0))

if __name__ == "__main__":
    unittest.main()
