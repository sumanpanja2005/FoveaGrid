import unittest
import numpy as np
from mapping.cell import FoveaCell
from mapping.pooling import risk_preserving_pool

class TestPooling(unittest.TestCase):
    def test_risk_preserving_obstacle_pooling(self):
        # 99 safe terrain cells + 1 obstacle cell
        children = [FoveaCell(terrain_prob=1.0, obstacle_flag=False) for _ in range(99)]
        hazard_cell = FoveaCell(
            static_prob=0.9,
            obstacle_flag=True,
            obstacle_height_max=2.5
        )
        children.append(hazard_cell)

        parent = risk_preserving_pool(children, target_resolution=0.50)

        # MANDATORY ASSERTIONS: Hazard must NOT disappear after merging!
        self.assertTrue(parent.obstacle_flag, "Error: Obstacle flag disappeared during cell merging!")
        self.assertEqual(parent.obstacle_height_max, 2.5, "Error: Obstacle height was diluted or lost!")
        self.assertTrue(parent.static_prob > 0.0)

    def test_risk_preserving_dynamic_pooling(self):
        # 99 static cells + 1 dynamic pedestrian cell
        children = [FoveaCell(static_prob=1.0, dynamic_flag=False) for _ in range(99)]
        dyn_cell = FoveaCell(
            dynamic_prob=0.95,
            dynamic_flag=True,
            velocity_x=1.4,
            velocity_y=0.0
        )
        children.append(dyn_cell)

        parent = risk_preserving_pool(children, target_resolution=0.50)

        self.assertTrue(parent.dynamic_flag, "Error: Dynamic flag disappeared during cell merging!")
        self.assertEqual(parent.velocity_x, 1.4)

    def test_risk_preserving_overhead_pooling(self):
        children = [FoveaCell(terrain_prob=1.0, overhang_flag=False) for _ in range(9)]
        overhang_cell = FoveaCell(
            overhead_prob=0.9,
            overhang_flag=True,
            obstacle_height_max=4.2
        )
        children.append(overhang_cell)

        parent = risk_preserving_pool(children, target_resolution=0.50)

        self.assertTrue(parent.overhang_flag, "Error: Overhead flag disappeared during cell merging!")
        self.assertEqual(parent.obstacle_height_max, 4.2)

if __name__ == "__main__":
    unittest.main()
