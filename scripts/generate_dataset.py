import os
import sys
import argparse
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import SyntheticEnvironment
from simulation.lidar_simulator import LidarSimulator
from pipeline.ingestion import PointCloudData, LidarDataIngestor

def generate_dataset(
    num_scenes: int = 2,
    scans_per_scene: int = 5,
    output_dir: str = "data/synthetic",
    export_formats: str = "npy"
):
    """
    Generates synthetic 64-beam automotive LiDAR dataset featuring:
    - Ground terrain with road elevation slope & roughness
    - Road curbs (15cm) and surface potholes (15-20cm)
    - Dynamic crossing pedestrians & oncoming vehicles
    - Static roadside infrastructure (utility poles & parked vehicles)
    - Low-clearance overhead obstacles
    - Exact 64-beam LiDAR ring IDs [0 - 63] & spherical noise/dropout modeling
    - Multi-format export: .npy, .pcd, .ply, .bin
    """
    formats = [f.strip().lower().lstrip(".") for f in export_formats.split(",") if f.strip()]
    if not formats:
        formats = ["npy"]

    print("=" * 80)
    print("        FOVEAGRID SYNTHETIC 64-BEAM LIDAR DATASET GENERATOR")
    print("=" * 80)
    print(f"Output Directory  : {output_dir}")
    print(f"Scenes to Generate: {num_scenes}")
    print(f"Scans per Scene   : {scans_per_scene} ({num_scenes * scans_per_scene} total scans)")
    print(f"Export Formats    : {', '.join(formats).upper()}")
    print("-" * 80)

    os.makedirs(output_dir, exist_ok=True)
    simulator = LidarSimulator(beams=64)

    total_scans_created = 0
    total_points_created = 0

    for scene_idx in range(1, num_scenes + 1):
        scene_dir = os.path.join(output_dir, f"scene_{scene_idx:06d}")
        os.makedirs(scene_dir, exist_ok=True)

        env = SyntheticEnvironment(scene_id=scene_idx, seed=42 + scene_idx)
        print(f"\n[Scene {scene_idx:02d}/{num_scenes:02d}] Initialized procedural 3D environment (Seed: {42 + scene_idx})")

        for scan_idx in range(scans_per_scene):
            dt = 0.1
            env.step(dt)
            scan_data = simulator.simulate_scan(env, timestamp=scan_idx * dt)

            pts = scan_data["points"]
            inten = scan_data["intensity"]
            rings = scan_data["ring"]
            labels = scan_data["labels"]
            times = scan_data["timestamp"]
            gt_boxes = scan_data["gt_boxes"]

            total_scans_created += 1
            total_points_created += len(pts)

            scan_base = os.path.join(scene_dir, f"scan_{scan_idx:03d}")

            # 1. Primary NumPy Dictionary (.npy)
            if "npy" in formats:
                np.save(f"{scan_base}.npy", scan_data)

            # Export standard point cloud formats if requested
            if any(f in formats for f in ["pcd", "ply", "bin"]):
                pcd = PointCloudData(
                    points=pts,
                    intensity=inten,
                    rings=rings,
                    timestamps=times,
                    labels=labels,
                    metadata={
                        "scene_id": scene_idx,
                        "scan_id": scan_idx,
                        "ego_pose": scan_data["ego_pose"].tolist(),
                        "num_gt_boxes": len(gt_boxes)
                    }
                )

                if "pcd" in formats:
                    LidarDataIngestor.save_pcd(pcd, f"{scan_base}.pcd", mode="ascii")
                if "ply" in formats:
                    LidarDataIngestor.save_ply(pcd, f"{scan_base}.ply", mode="binary")
                if "bin" in formats:
                    LidarDataIngestor.save_bin(pcd, f"{scan_base}.bin")

            ring_min = int(np.min(rings)) if len(rings) > 0 else 0
            ring_max = int(np.max(rings)) if len(rings) > 0 else 0
            print(
                f"  -> Scan {scan_idx:03d}: {len(pts):,} points | "
                f"Rings: [{ring_min:02d} - {ring_max:02d}] | "
                f"Objects: {len(gt_boxes)} | Saved: {scan_base}.[{','.join(formats)}]"
            )

    print("\n" + "=" * 80)
    print("           SYNTHETIC DATASET GENERATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print(f"Total Scans Saved : {total_scans_created}")
    print(f"Total 3D Points   : {total_points_created:,}")
    print(f"Destination       : {os.path.abspath(output_dir)}")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FoveaGrid Synthetic Dataset Generator")
    parser.add_argument("--scenes", type=int, default=2, help="Number of synthetic scenes (default: 2)")
    parser.add_argument("--scans", type=int, default=5, help="Scans per scene (default: 5)")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--export_formats", type=str, default="npy", help="Comma-separated formats: npy,pcd,ply,bin")
    args = parser.parse_args()

    generate_dataset(
        num_scenes=args.scenes,
        scans_per_scene=args.scans,
        output_dir=args.output,
        export_formats=args.export_formats
    )
