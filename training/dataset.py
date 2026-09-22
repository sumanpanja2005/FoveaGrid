import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

class LidarScanDataset(Dataset):
    """
    PyTorch Dataset loader for synthetic 64-beam LiDAR scans.
    Computes 11 normalized point features:
    [x, y, z, range, azimuth, elevation, intensity, ring, timestamp, local_density, height_above_ground]
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.scan_files = []

        if os.path.exists(data_dir):
            for root, _, files in os.walk(data_dir):
                for f in sorted(files):
                    if f.startswith("scan_") and f.endswith(".npy"):
                        self.scan_files.append(os.path.join(root, f))

    def __len__(self) -> int:
        return len(self.scan_files)

    def extract_features(self, points: np.ndarray, intensity: np.ndarray, ring: np.ndarray, timestamp: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.zeros((0, 11), dtype=np.float32)

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        r = np.linalg.norm(points, axis=1)
        azimuth = np.arctan2(y, x)
        elevation = np.arcsin(np.clip(z / np.maximum(r, 1e-6), -1.0, 1.0))

        # Local ground estimation approximation
        z_rel = z - np.min(z)

        # Fast local density estimation
        density = 1.0 / (1.0 + 0.1 * r)

        features = np.stack([
            x / 100.0,
            y / 100.0,
            z / 10.0,
            r / 100.0,
            azimuth / np.pi,
            elevation / (np.pi / 2.0),
            intensity,
            ring / 64.0,
            timestamp / 10.0,
            density,
            z_rel / 5.0
        ], axis=-1)

        return features.astype(np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        filepath = self.scan_files[idx]
        data = np.load(filepath, allow_pickle=True).item()

        pts = data["points"]
        intensity = data["intensity"]
        ring = data["ring"]
        timestamp = data["timestamp"]
        labels = np.clip(data["labels"], 0, 4)

        features = self.extract_features(pts, intensity, ring, timestamp)

        return {
            "points": torch.from_numpy(pts).float(),
            "features": torch.from_numpy(features).float(),
            "labels": torch.from_numpy(labels).long()
        }
