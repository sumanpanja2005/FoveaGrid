import numpy as np

def compute_cell_confidence(
    observation_count: int,
    distance_to_sensor: float,
    max_sensor_range: float = 100.0,
    semantic_certainty: float = 0.9,
    temporal_consistency: float = 1.0
) -> float:
    """
    Formulates range-aware cell confidence score:
    confidence = observation_factor * range_factor * semantic_certainty * temporal_consistency
    """
    # 1. Observation density factor (saturates at 10 observations)
    obs_factor = min(1.0, observation_count / 10.0)

    # 2. Distance decay factor (quadratic decay with distance)
    range_ratio = np.clip(distance_to_sensor / max_sensor_range, 0.0, 1.0)
    range_factor = 1.0 - 0.7 * (range_ratio ** 2)

    # 3. Combined confidence score
    confidence = obs_factor * range_factor * semantic_certainty * temporal_consistency
    return float(np.clip(confidence, 0.0, 1.0))
