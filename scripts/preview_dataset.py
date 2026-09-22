import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def preview_dataset(scan_path: str = "data/synthetic/scene_000001/scan_000.npy", save_fig_path: str = "docs/sample_scan_preview.png"):
    if not os.path.exists(scan_path):
        print(f"Scan file {scan_path} not found. Running generator first...")
        from scripts.generate_dataset import generate_dataset
        generate_dataset(num_scenes=2, scans_per_scene=5)

    data = np.load(scan_path, allow_pickle=True).item()
    pts = data["points"]
    intensity = data["intensity"]
    ring = data["ring"]
    timestamp = data["timestamp"]
    labels = data["labels"]
    gt_boxes = data["gt_boxes"]

    print("=" * 80)
    print(f"             FOVEAGRID SYNTHETIC DATASET SCAN PREVIEW")
    print("=" * 80)
    print(f"File Path: {scan_path}")
    print(f"Total Points: {len(pts):,} | 64-Beam Rings: [0 - {np.max(ring)}]")
    print(f"Coordinate Extents (X, Y, Z):")
    print(f"  - X range: [{np.min(pts[:, 0]):.2f}m, {np.max(pts[:, 0]):.2f}m]")
    print(f"  - Y range: [{np.min(pts[:, 1]):.2f}m, {np.max(pts[:, 1]):.2f}m]")
    print(f"  - Z range: [{np.min(pts[:, 2]):.2f}m, {np.max(pts[:, 2]):.2f}m]")

    class_names = {
        0: "Unknown", 1: "Terrain (Road/Grass)", 2: "Static Obstacle",
        3: "Dynamic Object", 4: "Overhead Obstacle", 5: "Curb", 6: "Pothole"
    }

    print("\n--- GROUND-TRUTH SEMANTIC CLASS BREAKDOWN ---")
    unique_lbls, counts = np.unique(labels, return_counts=True)
    for lbl, count in zip(unique_lbls, counts):
        cname = class_names.get(lbl, "Unknown")
        pct = (count / len(labels)) * 100.0
        print(f"  - Class {lbl} [{cname}]: {count:,} points ({pct:.1f}%)")

    print("\n--- GROUND-TRUTH 3D OBJECT BOUNDING BOXES ---")
    for box in gt_boxes:
        print(f"  - ID {box['id']} | Category: {box['category']} | Pos: {box['position']} | Size: {box['size']} | Vel: {box['velocity']}")

    # Generate visual PNG plot
    os.makedirs(os.path.dirname(save_fig_path), exist_ok=True)
    fig = plt.figure(figsize=(12, 10))
    
    ax = fig.add_subplot(111, projection='3d')
    colors = plt.cm.plasma(intensity)
    
    # Subsample for clear rendering
    sub_idx = np.random.choice(len(pts), size=min(15000, len(pts)), replace=False)
    ax.scatter(pts[sub_idx, 0], pts[sub_idx, 1], pts[sub_idx, 2], c=labels[sub_idx], cmap='viridis', s=1.5, alpha=0.8)

    ax.set_title("Synthetic 64-Beam LiDAR Point Cloud (Color-Coded by Semantic Label)", fontsize=14)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    plt.tight_layout()
    plt.savefig(save_fig_path, dpi=150)
    plt.close()
    print(f"\nSaved scan visual preview figure to {save_fig_path}")
    print("=" * 80)

if __name__ == "__main__":
    preview_dataset()
