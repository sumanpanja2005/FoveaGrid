import numpy as np
from typing import Dict, Any, Optional
from .dem_elevation import DEMGrid
from .bev_projection import BEVGrid
from .feature_extraction import DetectedFeatures

class Unified25DPayloadPacker:
    """
    Serializes 2.5D DEM raster elevation, curbs, potholes, 3D obstacle bounding boxes,
    and telemetry into an optimized WebSocket payload.
    """

    @classmethod
    def pack(
        cls,
        dem: DEMGrid,
        features: DetectedFeatures,
        bev: Optional[BEVGrid] = None,
        timings: Optional[Dict[str, float]] = None,
        raw_count: int = 0,
        processed_count: int = 0,
        ground_count: int = 0,
        source_name: str = "Synthetic",
        source_format: str = "Stream",
        subsample_dem_step: int = 2
    ) -> Dict[str, Any]:
        """
        Packs unified 2.5D map frame.
        """
        # Downsample DEM slightly for high-frequency streaming if resolution is very high
        step = max(1, subsample_dem_step)
        elev_sub = dem.elevation[::step, ::step]
        h_sub, w_sub = elev_sub.shape

        # Clean NaN/Inf values
        clean_elev = np.nan_to_num(elev_sub, nan=-2.0, posinf=10.0, neginf=-10.0)
        min_z = float(np.min(clean_elev))
        max_z = float(np.max(clean_elev))

        # Flatten elevation values rounded to 2 decimals for bandwidth efficiency
        flat_elevation = np.round(clean_elev.flatten(), 2).tolist()

        # 1. Road Curbs (subsample up to 400 points)
        curb_coords = []
        if len(features.curb_points) > 0:
            c_pts = features.curb_points
            if len(c_pts) > 400:
                idx = np.linspace(0, len(c_pts) - 1, 400).astype(int)
                c_pts = c_pts[idx]
            curb_coords = np.round(c_pts, 2).tolist()

        # 2. Potholes
        potholes = []
        for ph in features.potholes:
            potholes.append({
                "center": [round(v, 2) for v in ph["center"]],
                "radius": round(ph["radius"], 2),
                "max_depth": round(ph["max_depth"], 2),
                "point_count": ph.get("point_count", 0)
            })

        # 3. 3D Obstacle & Infrastructure Bounding Boxes
        obstacles = []
        all_clusters = features.dynamic_obstacles + features.roadside_infrastructure + features.static_obstacles
        for obs in all_clusters:
            obstacles.append({
                "id": obs.cluster_id,
                "category": obs.category,
                "position": [round(float(v), 2) for v in obs.position],
                "size": [round(float(v), 2) for v in obs.size],
                "linearity": round(float(obs.linearity), 2),
                "verticality": round(float(obs.verticality), 2)
            })

        # 4. Telemetry Metrics
        t_dict = timings or {}
        total_ms = t_dict.get("total_pipeline_ms", 50.0)
        fps = round(1000.0 / max(0.1, total_ms), 1)

        ground_pct = round((ground_count / max(1, processed_count)) * 100.0, 1)

        payload = {
            "type": "unified_25d_frame",
            "dem": {
                "width": w_sub,
                "height": h_sub,
                "bounds": [float(b) for b in dem.bounds],
                "resolution": round(dem.resolution * step, 2),
                "elevation": flat_elevation,
                "min_z": round(min_z, 2),
                "max_z": round(max_z, 2)
            },
            "curbs": curb_coords,
            "potholes": potholes,
            "obstacles": obstacles,
            "telemetry": {
                "source": source_name,
                "format": source_format,
                "raw_points": raw_count,
                "processed_points": processed_count,
                "ground_points": ground_count,
                "ground_pct": ground_pct,
                "timings_ms": {k: round(v, 2) for k, v in t_dict.items()},
                "fps": fps
            }
        }

        return payload
