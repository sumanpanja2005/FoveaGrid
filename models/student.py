import torch
import torch.nn as nn
from .sparse_unet import SparseUNet

class StudentSegmentationModel(nn.Module):
    """Lightweight Student Sparse U-Net Model (channels: [16, 32, 64, 32, 16]) for real-time inference."""

    def __init__(self, in_features: int = 11, num_classes: int = 5):
        super().__init__()
        self.net = SparseUNet(in_features=in_features, channels=[16, 32, 64, 32, 16], num_classes=num_classes)

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.net(points, features)
