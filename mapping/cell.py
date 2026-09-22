from dataclasses import dataclass, field
import numpy as np

@dataclass
class FoveaCell:
    """
    Compact 2.5D World-Anchored Grid Cell Representation:
    Ground heights, obstacle heights, semantic probabilities, risk flags, velocity vector, confidence.
    """
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    resolution: float = 0.05 # Cell spatial resolution (m)

    ground_height_min: float = 0.0
    ground_height_max: float = 0.0
    obstacle_height_min: float = 0.0
    obstacle_height_max: float = 0.0

    terrain_prob: float = 1.0
    static_prob: float = 0.0
    dynamic_prob: float = 0.0
    overhead_prob: float = 0.0

    obstacle_flag: bool = False
    dynamic_flag: bool = False
    overhang_flag: bool = False

    velocity_x: float = 0.0
    velocity_y: float = 0.0

    confidence: float = 0.0
    observation_count: int = 0
    last_update_timestamp: float = 0.0

    def get_log_odds(self) -> np.ndarray:
        """Returns log-odds vector for [terrain, static, dynamic, overhead]."""
        probs = np.clip([self.terrain_prob, self.static_prob, self.dynamic_prob, self.overhead_prob], 1e-4, 1.0 - 1e-4)
        return np.log(probs / (1.0 - probs))

    def update_from_log_odds(self, log_odds: np.ndarray):
        """Updates probabilities from log-odds vector."""
        probs = 1.0 / (1.0 + np.exp(-log_odds))
        probs = probs / np.sum(probs) # Normalize
        self.terrain_prob = float(probs[0])
        self.static_prob = float(probs[1])
        self.dynamic_prob = float(probs[2])
        self.overhead_prob = float(probs[3])
