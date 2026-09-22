"""
FoveaGrid Simulation Module: Procedural 3D Environment and 64-Beam LiDAR Simulator.
"""

from .terrain import SyntheticTerrain
from .objects import PhysicalObject, DynamicActor, StaticObstacle, OverheadObstacle, RoadHazard
from .environment import SyntheticEnvironment
from .noise_model import LidarNoiseModel
from .lidar_simulator import LidarSimulator

__all__ = [
    "SyntheticTerrain",
    "PhysicalObject",
    "DynamicActor",
    "StaticObstacle",
    "OverheadObstacle",
    "RoadHazard",
    "SyntheticEnvironment",
    "LidarNoiseModel",
    "LidarSimulator",
]
