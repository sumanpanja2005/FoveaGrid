import unittest
import numpy as np
from mapping.cell import FoveaCell
from mapping.fusion import TemporalProbabilisticFusion

class TestFusion(unittest.TestCase):
    def test_temporal_fusion_update(self):
        fusion = TemporalProbabilisticFusion()
        cell = FoveaCell(terrain_prob=0.5, static_prob=0.5)

        meas_probs = np.array([0.1, 0.9, 0.0, 0.0]) # Strong static observation
        fusion.fuse_cell(cell, meas_probs)

        self.assertTrue(cell.static_prob > 0.7)

    def test_dynamic_decay(self):
        fusion = TemporalProbabilisticFusion(decay_rate=0.8)
        cell = FoveaCell(dynamic_prob=0.8, dynamic_flag=True)

        # Static measurement over 5 frames
        meas_probs = np.array([0.9, 0.1, 0.0, 0.0])
        for _ in range(5):
            fusion.fuse_cell(cell, meas_probs, dt=0.1)

        self.assertFalse(cell.dynamic_flag) # Should decay and turn off

if __name__ == "__main__":
    unittest.main()
