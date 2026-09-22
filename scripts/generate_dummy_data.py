import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.ingestion import PointCloudData, LidarDataIngestor

def generate_dummy_data(output_dir: str = "data/dummy", num_points: int = 25000, seed: int = 42) -> str:
    """
    Generates realistic roadside 3D LiDAR point cloud data featuring:
    - Asphalt road pavement with road slope/crown
    - 15 cm concrete curbs along road borders
    - Sidewalk surface
    - 15-20 cm deep potholes / road surface depressions
    - Passing dynamic vehicles (sedan, SUV)
    - Crossing pedestrian
    - Roadside utility poles and trees
    
    Saves outputs in: .npy, .pcd, .ply, and .bin formats in output_dir.
    Returns path to primary sample file (.npy).
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    points = []
    intensities = []
    rings = []
    labels = []  # 1: terrain, 2: static, 3: dynamic, 4: overhead, 5: curb, 6: pothole

    # 1. Road Surface (40m forward/back, 10m wide)
    N_road = int(num_points * 0.72)
    rx = rng.uniform(-25.0, 35.0, N_road)
    ry = rng.uniform(-10.0, 10.0, N_road)
    
    # Base height: sensor at 1.75m elevation -> ground at z = -1.75m with subtle slope
    rz = -1.75 + 0.008 * rx + 0.012 * np.sin(0.15 * rx) * np.cos(0.25 * ry)
    road_labels = np.ones(N_road, dtype=np.int32)  # default terrain

    # Curbs (15 cm step at y = ±4.5m)
    is_curb = (np.abs(ry) >= 4.3) & (np.abs(ry) <= 4.7)
    rz[is_curb] += 0.15
    road_labels[is_curb] = 5

    # Sidewalks
    is_sidewalk = np.abs(ry) > 4.7
    rz[is_sidewalk] += 0.15

    # Potholes / surface depressions
    potholes = [
        (10.0, 1.2, 0.9, 0.18),    # Pothole 1: forward-right lane
        (-12.0, -1.8, 0.75, 0.15),  # Pothole 2: rear-left lane
        (22.0, -0.5, 1.1, 0.20)     # Pothole 3: forward lane center
    ]
    for px, py, pr, pdepth in potholes:
        d = np.hypot(rx - px, ry - py)
        in_p = d < pr
        if np.any(in_p):
            rz[in_p] -= pdepth * (1.0 - d[in_p] / pr)
            road_labels[in_p] = 6

    road_pts = np.column_stack([rx, ry, rz])
    road_inten = np.full(N_road, 0.35, dtype=np.float32) + rng.normal(0, 0.02, N_road)
    road_rings = np.clip(np.floor((rz + 2.0) / 0.1).astype(np.int32), 0, 63)

    points.append(road_pts)
    intensities.append(road_inten)
    rings.append(road_rings)
    labels.append(road_labels)

    # 2. Dynamic Vehicle #1 (Sedan at x = 14m, y = -2.5m)
    N_veh1 = int(num_points * 0.10)
    v1_x = rng.uniform(11.8, 16.2, N_veh1)
    v1_y = rng.uniform(-3.4, -1.6, N_veh1)
    v1_z = rng.uniform(-1.6, -0.2, N_veh1)
    points.append(np.column_stack([v1_x, v1_y, v1_z]))
    intensities.append(np.full(N_veh1, 0.85, dtype=np.float32))
    rings.append(np.random.randint(15, 45, N_veh1))
    labels.append(np.full(N_veh1, 3, dtype=np.int32))

    # 3. Dynamic Vehicle #2 (SUV at x = -10m, y = 2.4m)
    N_veh2 = int(num_points * 0.08)
    v2_x = rng.uniform(-12.3, -7.7, N_veh2)
    v2_y = rng.uniform(1.4, 3.4, N_veh2)
    v2_z = rng.uniform(-1.6, 0.0, N_veh2)
    points.append(np.column_stack([v2_x, v2_y, v2_z]))
    intensities.append(np.full(N_veh2, 0.80, dtype=np.float32))
    rings.append(np.random.randint(15, 50, N_veh2))
    labels.append(np.full(N_veh2, 3, dtype=np.int32))

    # 4. Crossing Pedestrian (at x = 6.0m, y = 1.0m)
    N_ped = int(num_points * 0.03)
    p_x = rng.uniform(5.7, 6.3, N_ped)
    p_y = rng.uniform(0.7, 1.3, N_ped)
    p_z = rng.uniform(-1.6, 0.1, N_ped)
    points.append(np.column_stack([p_x, p_y, p_z]))
    intensities.append(np.full(N_ped, 0.65, dtype=np.float32))
    rings.append(np.random.randint(20, 55, N_ped))
    labels.append(np.full(N_ped, 3, dtype=np.int32))

    # 5. Roadside Utility Poles & Trees
    for pole_x in [-16.0, 4.0, 24.0]:
        N_pole = int(num_points * 0.02)
        px_p = np.full(N_pole, pole_x) + rng.normal(0, 0.05, N_pole)
        py_p = np.full(N_pole, 5.5) + rng.normal(0, 0.05, N_pole)
        pz_p = np.linspace(-1.6, 2.5, N_pole)
        points.append(np.column_stack([px_p, py_p, pz_p]))
        intensities.append(np.full(N_pole, 0.90, dtype=np.float32))
        rings.append(np.random.randint(10, 60, N_pole))
        labels.append(np.full(N_pole, 2, dtype=np.int32))

    # Tree on left shoulder
    N_tree = int(num_points * 0.03)
    tx = -6.0 + rng.normal(0, 0.8, N_tree)
    ty = -6.5 + rng.normal(0, 0.8, N_tree)
    tz = rng.uniform(-1.5, 2.8, N_tree)
    points.append(np.column_stack([tx, ty, tz]))
    intensities.append(np.full(N_tree, 0.55, dtype=np.float32))
    rings.append(np.random.randint(10, 60, N_tree))
    labels.append(np.full(N_tree, 2, dtype=np.int32))

    all_pts = np.vstack(points).astype(np.float32)
    all_inten = np.clip(np.concatenate(intensities), 0.0, 1.0).astype(np.float32)
    all_rings = np.concatenate(rings).astype(np.int32)
    all_labels = np.concatenate(labels).astype(np.int32)

    pcd = PointCloudData(
        points=all_pts,
        intensity=all_inten,
        rings=all_rings,
        labels=all_labels,
        metadata={"format": "dummy_roadside", "num_points": len(all_pts)}
    )

    # Export to all standard formats
    npy_path = os.path.join(output_dir, "roadside_sample.npy")
    pcd_path = os.path.join(output_dir, "roadside_sample.pcd")
    ply_path = os.path.join(output_dir, "roadside_sample.ply")
    bin_path = os.path.join(output_dir, "roadside_sample.bin")

    # Save NPY dictionary
    np.save(npy_path, {
        "points": all_pts,
        "intensity": all_inten,
        "ring": all_rings,
        "labels": all_labels,
        "gt_boxes": [
            {"id": 1, "category": "vehicle", "position": [14.0, -2.5, -0.9], "size": [4.4, 1.8, 1.4]},
            {"id": 2, "category": "vehicle", "position": [-10.0, 2.4, -0.8], "size": [4.6, 2.0, 1.6]},
            {"id": 3, "category": "pedestrian", "position": [6.0, 1.0, -0.8], "size": [0.6, 0.6, 1.7]},
            {"id": 4, "category": "pole", "position": [4.0, 5.5, 0.45], "size": [0.3, 0.3, 4.1]}
        ]
    })

    # Save PCD
    LidarDataIngestor.save_pcd(pcd, pcd_path, mode="ascii")

    # Save PLY
    LidarDataIngestor.save_ply(pcd, ply_path, mode="binary")

    # Save BIN (KITTI float32 Nx4)
    LidarDataIngestor.save_bin(pcd, bin_path)

    print(f"[Dummy Generator] Successfully created realistic roadside LiDAR point cloud ({len(all_pts):,} points):")
    print(f"  - NumPy Array : {npy_path}")
    print(f"  - PCD Format  : {pcd_path}")
    print(f"  - PLY Format  : {ply_path}")
    print(f"  - BIN Format  : {bin_path}")

    return npy_path

if __name__ == "__main__":
    generate_dummy_data()
