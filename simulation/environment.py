import numpy as np
from typing import List, Tuple, Dict, Any
from .terrain import SyntheticTerrain
from .objects import PhysicalObject, DynamicActor, StaticObstacle, OverheadObstacle, RoadHazard

class SyntheticEnvironment:
    """
    Manages complete 3D synthetic environment state across sequential scans:
    - Terrain & ground profile
    - Static obstacles (walls, poles, trees, buildings)
    - Dynamic objects (pedestrians, cars, bicycles)
    - Overhead obstacles (bridges, tree branches)
    - Ego vehicle state (pose, velocity)
    """

    def __init__(self, scene_id: int = 1, seed: int = 42):
        self.scene_id = scene_id
        self.rng = np.random.default_rng(seed)

        self.terrain = SyntheticTerrain(road_width=10.0, seed=seed)
        self.objects: List[PhysicalObject] = []
        self.ego_pose = np.eye(4, dtype=np.float32) # Ego pose in World frame SE(3)
        self.ego_velocity = np.array([8.0, 0.0, 0.0], dtype=np.float32) # Default 8 m/s forward

        self._populate_scene()

    def _populate_scene(self):
        obj_id = 1

        # 1. Potholes & Curbs on Road
        self.terrain.add_pothole(x=25.0, y=1.2, radius=1.0, depth=0.20)
        self.terrain.add_pothole(x=45.0, y=-2.0, radius=0.8, depth=0.15)

        # 2. Static Obstacles (Walls, Poles, Parked Cars)
        # Parked Car on right shoulder
        self.objects.append(StaticObstacle(
            object_id=obj_id, position=np.array([18.0, -4.5, 0.8], dtype=np.float32),
            size=np.array([4.5, 1.8, 1.5], dtype=np.float32), heading=0.05
        ))
        obj_id += 1

        # Poles along sidewalk
        for x_pos in [10.0, 25.0, 40.0, 60.0, 80.0]:
            self.objects.append(StaticObstacle(
                object_id=obj_id, position=np.array([x_pos, 5.8, 2.5], dtype=np.float32),
                size=np.array([0.3, 0.3, 5.0], dtype=np.float32)
            ))
            obj_id += 1

        # 3. Dynamic Objects (Pedestrians, Oncoming Vehicle, Cyclist)
        # Pedestrian crossing at 35m
        self.objects.append(DynamicActor(
            object_id=obj_id, category="dynamic", class_id=3,
            position=np.array([35.0, 4.0, 0.85], dtype=np.float32),
            size=np.array([0.6, 0.6, 1.7], dtype=np.float32), heading=-np.pi/2,
            velocity=np.array([0.0, -1.4, 0.0], dtype=np.float32) # 1.4 m/s crossing speed
        ))
        obj_id += 1

        # Oncoming Vehicle at 60m
        self.objects.append(DynamicActor(
            object_id=obj_id, category="dynamic", class_id=3,
            position=np.array([60.0, 2.5, 0.8], dtype=np.float32),
            size=np.array([4.8, 2.0, 1.6], dtype=np.float32), heading=np.pi,
            velocity=np.array([-12.0, 0.0, 0.0], dtype=np.float32) # 12 m/s relative oncoming
        ))
        obj_id += 1

        # 4. Overhead Obstacle (Low bridge / ceiling at 30m)
        self.objects.append(OverheadObstacle(
            object_id=obj_id,
            position=np.array([30.0, 0.0, 4.2], dtype=np.float32), # z = 4.2m center
            size=np.array([6.0, 14.0, 1.5], dtype=np.float32) # Clearance ~3.45m
        ))
        obj_id += 1

    def step(self, dt: float = 0.1):
        """Advance time step for Ego vehicle and dynamic actors."""
        # 1. Update Ego Pose
        dx = self.ego_velocity[0] * dt
        dy = self.ego_velocity[1] * dt
        self.ego_pose[0, 3] += dx
        self.ego_pose[1, 3] += dy

        # 2. Update Ground elevation for Ego Z
        ego_x = self.ego_pose[0, 3]
        ego_y = self.ego_pose[1, 3]
        z_ground, _ = self.terrain.get_ground_elevation(np.array([ego_x]), np.array([ego_y]))
        self.ego_pose[2, 3] = z_ground[0] + 1.8 # LiDAR mounted 1.8m above ground

        # 3. Step Dynamic Objects
        for obj in self.objects:
            if isinstance(obj, DynamicActor):
                obj.step(dt)

    def get_ground_truth_boxes(self) -> List[Dict[str, Any]]:
        boxes = []
        for obj in self.objects:
            boxes.append({
                "id": obj.object_id,
                "class_id": obj.class_id,
                "category": obj.category,
                "position": obj.position.tolist(),
                "size": obj.size.tolist(),
                "heading": obj.heading,
                "velocity": obj.velocity.tolist()
            })
        return boxes
