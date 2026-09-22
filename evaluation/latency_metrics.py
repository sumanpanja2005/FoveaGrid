import time
from typing import Dict, Any

class LatencyProfiler:
    """
    Profiles execution latency across all pipeline stages in milliseconds and calculates pipeline FPS.
    """

    def __init__(self):
        self.stage_timings = {}
        self._start_times = {}

    def start_stage(self, stage_name: str):
        self._start_times[stage_name] = time.perf_counter()

    def end_stage(self, stage_name: str):
        if stage_name in self._start_times:
            elapsed_ms = (time.perf_counter() - self._start_times[stage_name]) * 1000.0
            self.stage_timings[stage_name] = elapsed_ms

    def get_summary(self) -> Dict[str, Any]:
        total_ms = sum(self.stage_timings.values())
        fps = 1000.0 / max(0.1, total_ms)
        return {
            "stage_ms": {k: round(v, 2) for k, v in self.stage_timings.items()},
            "total_ms": round(total_ms, 2),
            "fps": round(fps, 1)
        }
