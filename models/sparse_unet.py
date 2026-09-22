import torch
import torch.nn as nn
from typing import List
from .voxel_conv import VoxelSparseEngine

class SparseUNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class SparseUNet(nn.Module):
    """
    Sparse Point-Voxel U-Net Architecture for 3D LiDAR Semantic Segmentation.
    """

    def __init__(self, in_features: int = 11, channels: List[int] = [32, 64, 128, 64, 32], num_classes: int = 5):
        super().__init__()
        self.voxel_engine = VoxelSparseEngine()

        # Encoder stages
        self.enc1 = SparseUNetBlock(in_features, channels[0])
        self.enc2 = SparseUNetBlock(channels[0], channels[1])
        self.bottleneck = SparseUNetBlock(channels[1], channels[2])

        # Decoder stages
        self.dec2 = SparseUNetBlock(channels[2] + channels[1], channels[3])
        self.dec1 = SparseUNetBlock(channels[3] + channels[0], channels[4])

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(channels[4], channels[4]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(channels[4], num_classes)
        )

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Input:
        - points: (N, 3)
        - features: (N, in_features)
        
        Output:
        - logits: (N, num_classes)
        """
        if len(points) == 0:
            return torch.zeros(0, 5, device=points.device)

        # 1. Voxelization
        coords, vx_feat, p2v_map = self.voxel_engine.voxelize(points, features)

        # 2. Encoder pass
        e1 = self.enc1(vx_feat)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)

        # 3. Decoder pass with skip connections
        d2 = self.dec2(torch.cat([b, e2], dim=-1))
        d1 = self.dec1(torch.cat([d2, e1], dim=-1))

        # 4. Devoxelization to point level
        point_features = self.voxel_engine.devoxelize(d1, p2v_map)

        # 5. Classification prediction per point
        logits = self.classifier(point_features)
        return logits
