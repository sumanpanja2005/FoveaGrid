import unittest
import numpy as np
from mapping.ring_buffer import RingBufferIndex
from mapping.cell import FoveaCell

class TestGrid(unittest.TestCase):
    def test_ring_buffer_indexing(self):
        ring = RingBufferIndex(size_x=100, size_y=100, cell_size=0.1, origin_x=0.0, origin_y=0.0)
        
        ix, iy = ring.world_to_grid(0.05, 0.05)
        self.assertEqual(ix, 0)
        self.assertEqual(iy, 0)

        # Test shift origin
        ring.shift_origin(1.0, 1.0)
        self.assertEqual(ring.origin_x, 1.0)
        self.assertEqual(ring.origin_y, 1.0)

    def test_fovea_cell_log_odds(self):
        cell = FoveaCell(terrain_prob=0.8, static_prob=0.1, dynamic_prob=0.05, overhead_prob=0.05)
        log_odds = cell.get_log_odds()
        self.assertEqual(len(log_odds), 4)

        cell.update_from_log_odds(log_odds)
        self.assertAlmostEqual(cell.terrain_prob, 0.8, delta=0.05)

if __name__ == "__main__":
    unittest.main()
