import numpy as np
from typing import Tuple

class RingBufferIndex:
    """
    Toroidal ring-buffer spatial index for 2.5D world-anchored grid map.
    Allows continuous vehicle translation through the world without reallocating grid memory.
    """

    def __init__(self, size_x: int, size_y: int, cell_size: float = 0.05, origin_x: float = 0.0, origin_y: float = 0.0):
        self.size_x = size_x
        self.size_y = size_y
        self.cell_size = cell_size
        self.origin_x = origin_x
        self.origin_y = origin_y

        self.offset_x = 0
        self.offset_y = 0

    def world_to_grid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Converts world coordinates (x, y) to toroidal ring-buffer grid indices (ix, iy)."""
        gx = int(np.floor((wx - self.origin_x) / self.cell_size))
        gy = int(np.floor((wy - self.origin_y) / self.cell_size))

        ix = (gx + self.offset_x) % self.size_x
        iy = (gy + self.offset_y) % self.size_y
        return ix, iy

    def grid_to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        """Converts toroidal grid indices (ix, iy) to world center coordinates (wx, wy)."""
        gx = (ix - self.offset_x) % self.size_x
        gy = (iy - self.offset_y) % self.size_y

        wx = self.origin_x + (gx + 0.5) * self.cell_size
        wy = self.origin_y + (gy + 0.5) * self.cell_size
        return wx, wy

    def shift_origin(self, new_origin_x: float, new_origin_y: float):
        """Shifts toroidal origin as vehicle moves."""
        shift_x = int(np.round((new_origin_x - self.origin_x) / self.cell_size))
        shift_y = int(np.round((new_origin_y - self.origin_y) / self.cell_size))

        self.offset_x = (self.offset_x + shift_x) % self.size_x
        self.offset_y = (self.offset_y + shift_y) % self.size_y

        self.origin_x += shift_x * self.cell_size
        self.origin_y += shift_y * self.cell_size
