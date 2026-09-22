import os
import sys
import argparse
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.environment import SyntheticEnvironment
from simulation.lidar_simulator import LidarSimulator

def generate_dataset(num_scenes: int = 5, scans_per_scene: int = 10, output_dir: str = "data/synthetic"):
    print(f"[Dataset Generator] Generating {num_scenes} scenes with {scans_per_scene} sequential scans each...")
    os.makedirs(output_dir, exist_ok=True)

    simulator = LidarSimulator()

    for scene_idx in range(1, num_scenes + 1):
        scene_dir = os.path.join(output_dir, f"scene_{scene_idx:06d}")
        os.makedirs(scene_dir, exist_ok=True)

        env = SyntheticEnvironment(scene_id=scene_idx, seed=42 + scene_idx)

        for scan_idx in range(scans_per_scene):
            dt = 0.1
            env.step(dt)
            scan_data = simulator.simulate_scan(env, timestamp=scan_idx * dt)

            scan_path = os.path.join(scene_dir, f"scan_{scan_idx:03d}.npy")
            np.save(scan_path, scan_data)

    print(f"[Dataset Generator] Successfully generated dataset in {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FoveaGrid Synthetic Dataset Generator")
    parser.add_argument("--scenes", type=int, default=5, help="Number of synthetic scenes")
    parser.add_argument("--scans", type=int, default=10, help="Scans per scene")
    parser.add_argument("--output", type=str, default="data/synthetic", help="Output directory")
    args = parser.parse_args()

    generate_dataset(num_scenes=args.scenes, scans_per_scene=args.scans, output_dir=args.output)
