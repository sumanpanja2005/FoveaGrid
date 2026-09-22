import numpy as np
from .cell import FoveaCell

class TemporalProbabilisticFusion:
    """
    Probabilistic Log-Odds Temporal Fusion with Dynamic Decay:
    L_t = L_{t-1} + L_{measurement}
    Applies temporal decay to stale dynamic information so dynamic objects don't leave phantom trails.
    """

    def __init__(
        self,
        decay_rate: float = 0.95,
        max_log_odds: float = 4.0,
        min_log_odds: float = -4.0
    ):
        self.decay_rate = decay_rate
        self.max_log_odds = max_log_odds
        self.min_log_odds = min_log_odds

    def fuse_cell(self, cell: FoveaCell, measurement_probs: np.ndarray, dt: float = 0.1):
        """
        Fuses new measurement probabilities into existing cell using log-odds recursive update.
        """
        prior_log_odds = cell.get_log_odds()

        # Measurement log-odds
        meas_p = np.clip(measurement_probs, 1e-4, 1.0 - 1e-4)
        meas_log_odds = np.log(meas_p / (1.0 - meas_p))

        # Recursive update
        updated_log_odds = prior_log_odds + meas_log_odds

        # Clamp log odds
        updated_log_odds = np.clip(updated_log_odds, self.min_log_odds, self.max_log_odds)
        cell.update_from_log_odds(updated_log_odds)

        # Dynamic decay for dynamic probability if no new observation
        if measurement_probs[2] < 0.3 and cell.dynamic_flag:
            cell.dynamic_prob *= (self.decay_rate ** (dt / 0.1))
            if cell.dynamic_prob < 0.1:
                cell.dynamic_flag = False
