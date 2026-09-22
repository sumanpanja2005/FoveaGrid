import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class PhysicalObject:
    object_id: int
    category: str # "static", "dynamic", "overhead", "hazard"
    class_id: int # 1: terrain, 2: static, 3: dynamic, 4: overhead, 5: curb, 6: pothole
    position: np.ndarray # [x, y, z] center
    size: np.ndarray # [dx, dy, dz] extents
    heading: float = 0.0 # radians
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def get_bounding_box_corners(self) -> np.ndarray:
        """
        Returns 8 corners of 3D oriented bounding box.
        """
        dx, dy, dz = self.size / 2.0
        corners_local = np.array([
            [-dx, -dy, -dz],
            [ dx, -dy, -dz],
            [ dx,  dy, -dz],
            [-dx,  dy, -dz],
            [-dx, -dy,  dz],
            [ dx, -dy,  dz],
            [ dx,  dy,  dz],
            [-dx,  dy,  dz]
        ], dtype=np.float32)

        cos_h = np.cos(self.heading)
        sin_h = np.sin(self.heading)
        rot_z = np.array([
            [cos_h, -sin_h, 0],
            [sin_h,  cos_h, 0],
            [0,      0,     1]
        ], dtype=np.float32)

        corners_world = (rot_z @ corners_local.T).T + self.position
        return corners_world

    def ray_intersection(self, ray_origins: np.ndarray, ray_dirs: np.ndarray) -> np.ndarray:
        """
        Calculates exact analytical ray-box intersection distances for batch of rays.
        Returns distance array (np.inf for no hit).
        """
        # Transform rays into object local canonical frame
        cos_h = np.cos(-self.heading)
        sin_h = np.sin(-self.heading)
        rot_z_inv = np.array([
            [cos_h, -sin_h, 0],
            [sin_h,  cos_h, 0],
            [0,      0,     1]
        ], dtype=np.float32)

        origins_local = (rot_z_inv @ (ray_origins - self.position).T).T
        dirs_local = (rot_z_inv @ ray_dirs.T).T

        half_size = self.size / 2.0
        box_min = -half_size
        box_max =  half_size

        safe_dirs = np.where(np.abs(dirs_local) < 1e-8, 1.0, dirs_local)
        inv_dirs = np.where(np.abs(dirs_local) < 1e-8, 1e8, 1.0 / safe_dirs)

        t1 = (box_min - origins_local) * inv_dirs
        t2 = (box_max - origins_local) * inv_dirs

        t_min = np.maximum(np.maximum(np.minimum(t1[:, 0], t2[:, 0]),
                                      np.minimum(t1[:, 1], t2[:, 1])),
                           np.minimum(t1[:, 2], t2[:, 2]))

        t_max = np.minimum(np.minimum(np.maximum(t1[:, 0], t2[:, 0]),
                                      np.maximum(t1[:, 1], t2[:, 1])),
                           np.maximum(t1[:, 2], t2[:, 2]))

        hits = (t_max >= np.maximum(0.0, t_min))
        distances = np.where(hits, np.maximum(0.0, t_min), np.inf)
        return distances

class DynamicActor(PhysicalObject):
    def step(self, dt: float):
        """Advance trajectory by velocity * dt."""
        self.position += self.velocity * dt

class StaticObstacle(PhysicalObject):
    def __init__(self, object_id: int, position: np.ndarray, size: np.ndarray, heading: float = 0.0):
        super().__init__(
            object_id=object_id,
            category="static",
            class_id=2, # static obstacle
            position=position,
            size=size,
            heading=heading,
            velocity=np.zeros(3, dtype=np.float32)
        )

class OverheadObstacle(PhysicalObject):
    """Overhead bridges, low branches, hanging structures (z > clearance threshold)."""
    def __init__(self, object_id: int, position: np.ndarray, size: np.ndarray, heading: float = 0.0):
        super().__init__(
            object_id=object_id,
            category="overhead",
            class_id=4, # overhead obstacle
            position=position,
            size=size,
            heading=heading,
            velocity=np.zeros(3, dtype=np.float32)
        )

class RoadHazard(PhysicalObject):
    """Potholes, curbs, speed bumps, debris."""
    def __init__(self, object_id: int, category_type: str, position: np.ndarray, size: np.ndarray):
        class_id = 6 if category_type == "pothole" else (5 if category_type == "curb" else 2)
        super().__init__(
            object_id=object_id,
            category="hazard",
            class_id=class_id,
            position=position,
            size=size,
            heading=0.0,
            velocity=np.zeros(3, dtype=np.float32)
        )
