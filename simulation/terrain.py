import numpy as np
from typing import Tuple, Dict, Any

class SyntheticTerrain:
    """
    Generates procedural 3D terrain elevation and semantic surface properties.
    Supports:
    - Flat road
    - Sloped & uneven roads
    - Sidewalks & Curbs (15 cm step height)
    - Potholes (depressions z < ground) & Speed bumps
    - Grass, dirt, gravel surfaces
    """

    def __init__(
        self,
        road_width: float = 8.0,
        slope: float = 0.0,
        roughness: float = 0.02,
        seed: int = 42
    ):
        self.road_width = road_width
        self.slope = slope
        self.roughness = roughness
        self.rng = np.random.default_rng(seed)
        
        # Potholes list: [(x, y, radius, depth), ...]
        self.potholes = []
        # Curbs list: [(y_left, height), (y_right, height)]
        self.curb_height = 0.15 # 15 cm step
        self.sidewalk_width = 3.0

    def add_pothole(self, x: float, y: float, radius: float = 0.8, depth: float = 0.15):
        self.potholes.append((x, y, radius, depth))

    def get_ground_elevation(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates ground z elevation and fine semantic label (1: terrain, 5: curb, 6: pothole).
        """
        x_arr = np.asarray(x, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        # Base elevation with slope along X axis
        z = self.slope * x_arr

        # Default terrain label = 1
        labels = np.ones_like(x_arr, dtype=np.int32)

        # Perlin-like roughness noise
        z += self.roughness * np.sin(0.5 * x_arr) * np.cos(0.5 * y_arr)

        # Sidewalk & Curb calculation
        half_w = self.road_width / 2.0
        abs_y = np.abs(y_arr)
        
        # Sidewalk elevation step
        is_sidewalk = abs_y > half_w
        is_curb = (abs_y >= half_w - 0.2) & (abs_y <= half_w + 0.2)
        
        z[is_sidewalk] += self.curb_height
        labels[is_curb] = 5 # Curb label

        # Potholes: circular Gaussian/parabolic depressions
        for px, py, pr, pd in self.potholes:
            dist_sq = (x_arr - px)**2 + (y_arr - py)**2
            in_ph = dist_sq < (pr**2)
            if np.any(in_ph):
                # Parabolic depression
                depression = pd * (1.0 - dist_sq[in_ph] / (pr**2))
                z[in_ph] -= depression
                labels[in_ph] = 6 # Pothole label

        return z, labels
