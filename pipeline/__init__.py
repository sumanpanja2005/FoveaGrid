"""
FoveaGrid End-to-End Roadside 3D LiDAR Perception & 2.5D Elevation Mapping Pipeline
"""

from .ingestion import LidarDataIngestor, PointCloudData, SyntheticStreamGenerator
from .preprocessing import (
    voxel_downsample,
    statistical_outlier_removal,
    segment_ground_ransac,
    segment_ground_pmf
)
from .feature_extraction import (
    RoadFeatureExtractor,
    DetectedFeatures,
    ObstacleCluster
)
from .bev_projection import BEVProjector, BEVGrid
from .dem_elevation import DEMGenerator, DEMGrid
from .renderer_25d import ElevationRenderer25D
from .pipeline_runner import LidarPerceptionPipeline
from .payload import Unified25DPayloadPacker

__all__ = [
    "LidarDataIngestor",
    "PointCloudData",
    "SyntheticStreamGenerator",
    "voxel_downsample",
    "statistical_outlier_removal",
    "segment_ground_ransac",
    "segment_ground_pmf",
    "RoadFeatureExtractor",
    "DetectedFeatures",
    "ObstacleCluster",
    "BEVProjector",
    "BEVGrid",
    "DEMGenerator",
    "DEMGrid",
    "ElevationRenderer25D",
    "LidarPerceptionPipeline",
    "Unified25DPayloadPacker"
]
