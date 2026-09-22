import numpy as np

class LidarNoiseModel:
    """
    Realistic LiDAR Noise Model:
    - Distance-dependent range Gaussian noise
    - Angular jitter noise
    - Reflectivity & range-dependent intensity noise
    - Distance-dependent dropout / missing return probability
    - Motion distortion during scan rotation
    """

    def __init__(
        self,
        range_sigma_base: float = 0.02,
        range_sigma_far: float = 0.001,
        angular_sigma_deg: float = 0.03,
        dropout_prob_base: float = 0.01,
        dropout_prob_far_mult: float = 0.0003,
        seed: int = 42
    ):
        self.range_sigma_base = range_sigma_base
        self.range_sigma_far = range_sigma_far
        self.angular_sigma_rad = np.radians(angular_sigma_deg)
        self.dropout_prob_base = dropout_prob_base
        self.dropout_prob_far_mult = dropout_prob_far_mult
        self.rng = np.random.default_rng(seed)

    def apply_noise(
        self,
        points: np.ndarray, # Nx3 [x, y, z]
        intensities: np.ndarray, # N
        labels: np.ndarray # N
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(points) == 0:
            return points, intensities, labels

        # Convert to spherical coordinates: r, azimuth, elevation
        ranges = np.linalg.norm(points, axis=1)
        azimuths = np.arctan2(points[:, 1], points[:, 0])
        elevations = np.arcsin(np.clip(points[:, 2] / np.maximum(ranges, 1e-6), -1.0, 1.0))

        # 1. Distance-dependent range noise
        range_sigma = self.range_sigma_base + self.range_sigma_far * ranges
        range_noise = self.rng.normal(0.0, range_sigma, size=len(ranges))
        noisy_ranges = np.maximum(0.1, ranges + range_noise)

        # 2. Angular noise
        azimuth_noise = self.rng.normal(0.0, self.angular_sigma_rad, size=len(azimuths))
        elevation_noise = self.rng.normal(0.0, self.angular_sigma_rad, size=len(elevations))
        noisy_azimuths = azimuths + azimuth_noise
        noisy_elevations = np.clip(elevations + elevation_noise, -np.pi/2 + 0.01, np.pi/2 - 0.01)

        # Reconstruct 3D Cartesian coordinates
        cos_el = np.cos(noisy_elevations)
        noisy_x = noisy_ranges * cos_el * np.cos(noisy_azimuths)
        noisy_y = noisy_ranges * cos_el * np.sin(noisy_azimuths)
        noisy_z = noisy_ranges * np.sin(noisy_elevations)

        noisy_points = np.stack([noisy_x, noisy_y, noisy_z], axis=-1)

        # 3. Distance-dependent intensity degradation
        intensity_decay = 1.0 / (1.0 + 0.005 * ranges**2)
        noisy_intensities = np.clip(intensities * intensity_decay + self.rng.normal(0.0, 0.05, size=len(ranges)), 0.0, 1.0)

        # 4. Dropout mask (farther points drop out more frequently)
        dropout_probs = self.dropout_prob_base + self.dropout_prob_far_mult * (ranges**2)
        keep_mask = self.rng.random(size=len(ranges)) > dropout_probs

        return noisy_points[keep_mask], noisy_intensities[keep_mask], labels[keep_mask]
