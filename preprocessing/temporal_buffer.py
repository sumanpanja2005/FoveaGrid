import numpy as np
from collections import deque
from typing import List, Dict, Any

class SequentialScanBuffer:
    """
    Maintains sequential scans [t-2, t-1, t, t+1, t+2] and transforms them into common frame.
    """

    def __init__(self, capacity: int = 5):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add_scan(self, scan_data: Dict[str, Any]):
        self.buffer.append(scan_data)

    def is_full(self) -> bool:
        return len(self.buffer) == self.capacity

    def get_latest_scan(self) -> Dict[str, Any]:
        return self.buffer[-1] if len(self.buffer) > 0 else None

    def get_scans(self) -> List[Dict[str, Any]]:
        return list(self.buffer)

    def clear(self):
        self.buffer.clear()
