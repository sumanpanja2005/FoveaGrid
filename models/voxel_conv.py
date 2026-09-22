import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict

class VoxelSparseEngine(nn.Module):
    """
    Pure PyTorch Voxel Sparse Feature Extraction Engine:
    1. Sparse 3D Hash Voxelization
    2. Point-to-Voxel feature scatter pooling
    3. Multi-layer Voxel Dense/Sparse 1D/3D Convolutions
    4. Voxel-to-Point Un-voxelization interpolation
    """

    def __init__(self, voxel_size: Tuple[float, float, float] = (0.1, 0.1, 0.15)):
        super().__init__()
        self.voxel_size = torch.tensor(voxel_size, dtype=torch.float32)

    def voxelize(self, points: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input:
        - points: (N, 3) tensor
        - features: (N, F) tensor
        
        Returns:
        - voxel_coords: (M, 3) unique voxel grid indices
        - voxel_features: (M, F) aggregated voxel features
        - point_to_voxel_map: (N,) mapping from point to voxel index
        """
        device = points.device
        vx_size = self.voxel_size.to(device)

        # Compute voxel integer coordinates
        coords = torch.floor(points / vx_size).to(torch.int64)

        # Unique voxel coordinate extraction using hash mapping
        unique_coords, point_to_voxel_map = torch.unique(coords, dim=0, return_inverse=True)
        M = unique_coords.shape[0]

        # Aggregate point features per voxel using scatter max/mean pooling
        F = features.shape[1]
        voxel_features = torch.zeros(M, F, device=device)
        counts = torch.zeros(M, 1, device=device)

        voxel_features.scatter_add_(0, point_to_voxel_map.unsqueeze(1).expand(-1, F), features)
        counts.scatter_add_(0, point_to_voxel_map.unsqueeze(1), torch.ones_like(point_to_voxel_map, dtype=torch.float32).unsqueeze(1))

        voxel_features = voxel_features / torch.clamp(counts, min=1.0)
        return unique_coords, voxel_features, point_to_voxel_map

    def devoxelize(self, voxel_features: torch.Tensor, point_to_voxel_map: torch.Tensor) -> torch.Tensor:
        """Interpolates voxel features back to individual points."""
        return voxel_features[point_to_voxel_map]
