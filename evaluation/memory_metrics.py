from typing import Dict, Any

def evaluate_memory_metrics(fovea_grid, uniform_grid) -> Dict[str, Any]:
    """
    Evaluates memory consumption, active cell counts, and memory compression ratio:
    Compression Ratio = Uniform Grid Memory / FoveaGrid Memory
    """
    fovea_cells = fovea_grid.get_cell_count()
    fovea_bytes = fovea_grid.get_memory_bytes()
    fovea_mb = fovea_bytes / (1024.0 * 1024.0)

    uniform_cells = uniform_grid.get_cell_count()
    uniform_bytes = uniform_grid.get_memory_bytes()
    uniform_mb = uniform_bytes / (1024.0 * 1024.0)

    compression_ratio = float(uniform_bytes / max(1, fovea_bytes))

    return {
        "fovea_cells": fovea_cells,
        "fovea_memory_mb": round(fovea_mb, 2),
        "uniform_cells": uniform_cells,
        "uniform_memory_mb": round(uniform_mb, 2),
        "compression_ratio": round(compression_ratio, 2)
    }
