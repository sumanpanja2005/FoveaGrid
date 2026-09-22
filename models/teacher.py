import torch
import torch.nn as nn
from .sparse_unet import SparseUNet

class TeacherSegmentationModel(nn.Module):
    """High-capacity Teacher Sparse U-Net Model (channels: [32, 64, 128, 64, 32])."""

    def __init__(self, in_features: int = 11, num_classes: int = 5):
        super().__init__()
        self.net = SparseUNet(in_features=in_features, channels=[32, 64, 128, 64, 32], num_classes=num_classes)

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.net(points, features)
