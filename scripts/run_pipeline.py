import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.pipeline_runner import LidarPerceptionPipeline
from pipeline.ingestion import LidarDataIngestor, PointCloudData

def main():
    parser = argparse.ArgumentParser(description="End-to-End Roadside 3D LiDAR Perception & 2.5D Elevation Mapping Pipeline")
    parser.add_argument("--input", type=str, default="data/dummy/roadside_sample.npy", help="Input LiDAR file (.pcd, .las, .ply, .bin, .npy)")
    parser.add_argument("--output", type=str, default="docs/pipeline_25d_elevation_demo.png", help="Path to save rendered visualization")
    parser.add_argument("--ground_method", type=str, default="ransac", choices=["ransac", "pmf"], help="Ground plane segmentation method")
    parser.add_argument("--voxel_size", type=float, default=0.15, help="Voxel downsampling resolution in meters")
    parser.add_argument("--dem_res", type=float, default=0.25, help="DEM 2.5D grid resolution in meters")
    parser.add_argument("--bev_res", type=float, default=0.20, help="BEV grid resolution in meters")
    parser.add_argument("--mode", type=str, default="dashboard", choices=["dashboard", "isometric"], help="Visualization mode")
    parser.add_argument("--interactive", action="store_true", help="Display interactive window")
    args = parser.parse_args()

    print("=" * 80)
    print("      AUTONOMOUS 3D LIDAR DATA PROCESSING & 2.5D ELEVATION MAPPING")
    print("=" * 80)
    print(f"Input Point Cloud : {args.input}")
    print(f"Ground Method     : {args.ground_method.upper()}")
    print(f"Voxel Resolution  : {args.voxel_size} m")
    print(f"DEM 2.5D Grid Res : {args.dem_res} m")
    print(f"Render Mode       : {args.mode}")

    # Fallback to dummy data generator if file doesn't exist
    if not os.path.exists(args.input):
        print(f"\n[Ingest] Input file '{args.input}' not found. Generating dummy roadside LiDAR scan...")
        from scripts.generate_dummy_data import generate_dummy_data
        generate_dummy_data(output_dir="data/dummy")
        args.input = "data/dummy/roadside_sample.npy"

    pipeline = LidarPerceptionPipeline(
        voxel_size=args.voxel_size,
        ground_method=args.ground_method,
        bev_resolution=args.bev_res,
        dem_resolution=args.dem_res
    )

    print("\n[Pipeline] Running end-to-end perception workflow...")
    result = pipeline.process(args.input)

    print("\n--- 1. POINT CLOUD & SEGMENTATION SUMMARY ---")
    print(f"Raw Input Points       : {result['raw_points_count']:,}")
    print(f"Processed / Filtered   : {result['processed_points_count']:,} (downsampled to {args.voxel_size}m)")
    print(f"Ground Surface Points  : {result['ground_points_count']:,} ({result['ground_points_count']/max(1, result['processed_points_count'])*100:.1f}%)")
    print(f"Obstacle / Object Pts  : {result['obstacle_points_count']:,}")
    if result['ground_plane']:
        a, b, c, d = result['ground_plane']
        print(f"RANSAC Ground Plane Eq : {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

    features = result['features']
    print("\n--- 2. DETECTED ROAD & ROADSIDE FEATURES ---")
    print(f"Detected Curb Points   : {len(features.curb_points):,}")
    print(f"Detected Potholes      : {len(features.potholes)}")
    for i, ph in enumerate(features.potholes, 1):
        print(f"   [{i}] Center: [{ph['center'][0]:.1f}, {ph['center'][1]:.1f}, {ph['center'][2]:.2f}] | Max Depth: {ph['max_depth']:.2f} m | Radius: {ph['radius']:.2f} m")

    print(f"Dynamic Obstacles      : {len(features.dynamic_obstacles)}")
    for obs in features.dynamic_obstacles:
        print(f"   - {obs.category.upper()} at [{obs.position[0]:.1f}, {obs.position[1]:.1f}, {obs.position[2]:.1f}] | Dimensions: [{obs.size[0]:.1f}x{obs.size[1]:.1f}x{obs.size[2]:.1f}m]")

    print(f"Roadside Infrastructure: {len(features.roadside_infrastructure)}")
    for infra in features.roadside_infrastructure:
        print(f"   - {infra.category.upper()} at [{infra.position[0]:.1f}, {infra.position[1]:.1f}, {infra.position[2]:.1f}] | Linearity: {infra.linearity:.2f} | Verticality: {infra.verticality:.2f}")

    print("\n--- 3. 2D BEV & 2.5D DEM GRID PROPERTIES ---")
    bev = result['bev']
    dem = result['dem']
    print(f"BEV Multi-Channel Grid : {bev.tensor.shape} (Channels: Occupancy, Zmax, Zmin, Delta_Z, Intensity, Density)")
    print(f"DEM 2.5D Raster Height : {dem.elevation.shape} | Elevation Range: [{np.nanmin(dem.elevation):.2f}m to {np.nanmax(dem.elevation):.2f}m]")

    print("\n--- 4. PIPELINE LATENCY PROFILING ---")
    for stage, ms in result['timings'].items():
        print(f"   - {stage:<25} : {ms:.2f} ms")
    print(f"Throughput : {result['fps']} FPS")

    print(f"\n[Renderer] Generating {args.mode.upper()} visualization with false-color gradient...")
    pipeline.render(
        result,
        output_image_path=args.output,
        dashboard=(args.mode == "dashboard"),
        show_interactive=args.interactive
    )

    print("\n" + "=" * 80)
    print("               PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
