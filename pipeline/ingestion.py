import os
import struct
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union, List, Tuple

@dataclass
class PointCloudData:
    """Standardized Container for 3D LiDAR Point Cloud Data."""
    points: np.ndarray  # Shape: (N, 3) float32 [x, y, z]
    intensity: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))  # Shape: (N,) [0..1]
    rings: Optional[np.ndarray] = None  # Shape: (N,) int32
    timestamps: Optional[np.ndarray] = None  # Shape: (N,) float32
    labels: Optional[np.ndarray] = None  # Ground truth or predicted labels (N,) int32
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.points)

class LidarDataIngestor:
    """
    Universal Zero-Dependency Ingestor for 3D LiDAR Datasets:
    Supports .pcd, .las, .ply, .bin (KITTI), .npy/.npz, and ROS-like PointCloud2 structures.
    """

    @classmethod
    def load(cls, file_or_data: Union[str, Dict[str, Any], np.ndarray]) -> PointCloudData:
        """
        Dispatches loading according to file extension, magic header, or object type.
        """
        if isinstance(file_or_data, str):
            ext = os.path.splitext(file_or_data)[1].lower()
            if ext == ".pcd":
                return cls.load_pcd(file_or_data)
            elif ext in [".las", ".laz"]:
                return cls.load_las(file_or_data)
            elif ext == ".ply":
                return cls.load_ply(file_or_data)
            elif ext == ".bin":
                return cls.load_bin(file_or_data)
            elif ext in [".npy", ".npz"]:
                return cls.load_npy(file_or_data)
            else:
                # Attempt auto-detection via header inspection
                return cls.auto_detect_file(file_or_data)
        elif isinstance(file_or_data, dict):
            return cls.load_dict_or_ros(file_or_data)
        elif isinstance(file_or_data, np.ndarray):
            pts = file_or_data[:, :3].astype(np.float32)
            intensity = file_or_data[:, 3].astype(np.float32) if file_or_data.shape[1] > 3 else np.zeros(len(pts), dtype=np.float32)
            if np.max(intensity) > 1.0:
                intensity = intensity / max(255.0, float(np.max(intensity)))
            return PointCloudData(points=pts, intensity=intensity, metadata={"format": "ndarray"})
        else:
            raise TypeError(f"Cannot ingest point cloud from type {type(file_or_data)}")

    @classmethod
    def auto_detect_file(cls, filepath: str) -> PointCloudData:
        """Inspects file header to auto-detect format."""
        with open(filepath, 'rb') as f:
            magic = f.read(12)
        if magic.startswith(b'# .PCD'):
            return cls.load_pcd(filepath)
        elif magic.startswith(b'LASF'):
            return cls.load_las(filepath)
        elif magic.startswith(b'ply'):
            return cls.load_ply(filepath)
        elif magic.startswith(b'\x93NUMPY'):
            return cls.load_npy(filepath)
        else:
            # Fallback to binary float stream
            return cls.load_bin(filepath)

    @classmethod
    def load_pcd(cls, filepath: str) -> PointCloudData:
        """Parses PCD file format (ASCII and Binary)."""
        with open(filepath, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline().decode('ascii', errors='ignore').strip()
                header_lines.append(line)
                if line.startswith('DATA'):
                    break

            header = {}
            for h in header_lines:
                tokens = h.split()
                if not tokens or tokens[0].startswith('#'):
                    continue
                header[tokens[0].upper()] = tokens[1:]

            data_type = header.get('DATA', ['ascii'])[0].lower()
            fields = header.get('FIELDS', ['x', 'y', 'z'])
            sizes = [int(s) for s in header.get('SIZE', ['4'] * len(fields))]
            types = header.get('TYPE', ['F'] * len(fields))
            points_count = int(header.get('POINTS', [header.get('WIDTH', ['0'])[0]])[0])

            field_idx = {name.lower(): idx for idx, name in enumerate(fields)}
            x_idx = field_idx.get('x', 0)
            y_idx = field_idx.get('y', 1)
            z_idx = field_idx.get('z', 2)
            i_idx = field_idx.get('intensity', field_idx.get('i', -1))
            ring_idx = field_idx.get('ring', -1)

            if data_type == 'ascii':
                content = f.read().decode('ascii', errors='ignore').strip().split('\n')
                records = []
                for row in content:
                    tokens = row.strip().split()
                    if len(tokens) >= 3:
                        records.append([float(val) for val in tokens])
                data_mat = np.array(records, dtype=np.float32)
            elif data_type == 'binary':
                type_map = {'F': 'f', 'U': 'u', 'I': 'i'}
                dtype_list = [(name, f"{type_map.get(t.upper(), 'f')}{s}") for name, s, t in zip(fields, sizes, types)]
                np_dtype = np.dtype(dtype_list)
                raw_bytes = f.read()
                data_rec = np.frombuffer(raw_bytes, dtype=np_dtype, count=points_count)
                data_mat = np.column_stack([data_rec[name].astype(np.float32) for name in fields])
            else:
                raise NotImplementedError(f"PCD data mode '{data_type}' is not supported.")

            pts = data_mat[:, [x_idx, y_idx, z_idx]]
            intensities = data_mat[:, i_idx] if i_idx >= 0 else np.zeros(len(pts), dtype=np.float32)
            if len(intensities) > 0 and np.max(intensities) > 1.0:
                intensities = intensities / max(255.0, float(np.max(intensities)))
            rings = data_mat[:, ring_idx].astype(np.int32) if ring_idx >= 0 else None

            return PointCloudData(
                points=pts,
                intensity=intensities.astype(np.float32),
                rings=rings,
                metadata={"format": "pcd", "data_type": data_type, "path": filepath}
            )

    @classmethod
    def save_pcd(cls, pcd_data: PointCloudData, filepath: str, mode: str = "ascii"):
        """Exports PointCloudData to a valid .pcd file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        pts = pcd_data.points
        N = len(pts)
        intensity = pcd_data.intensity if len(pcd_data.intensity) == N else np.zeros(N, dtype=np.float32)

        header = (
            "# .PCD v0.7 - Point Cloud Data\n"
            "VERSION 0.7\n"
            "FIELDS x y z intensity\n"
            "SIZE 4 4 4 4\n"
            "TYPE F F F F\n"
            "COUNT 1 1 1 1\n"
            f"WIDTH {N}\n"
            "HEIGHT 1\n"
            "VIEWPOINT 0 0 0 1 0 0 0\n"
            f"POINTS {N}\n"
            f"DATA {mode}\n"
        )
        with open(filepath, 'wb') as f:
            f.write(header.encode('ascii'))
            if mode == 'ascii':
                for i in range(N):
                    f.write(f"{pts[i, 0]:.4f} {pts[i, 1]:.4f} {pts[i, 2]:.4f} {intensity[i]:.4f}\n".encode('ascii'))
            else:
                buf = np.column_stack([pts, intensity]).astype(np.float32).tobytes()
                f.write(buf)

    @classmethod
    def load_ply(cls, filepath: str) -> PointCloudData:
        """Parses Polygon File Format (PLY) for point clouds."""
        with open(filepath, 'rb') as f:
            header_lines = []
            while True:
                line = f.readline().decode('ascii', errors='ignore').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break

            format_line = next((l for l in header_lines if l.startswith('format')), 'format ascii 1.0')
            is_binary = 'binary_little_endian' in format_line
            
            num_vertices = 0
            properties = []
            reading_vertex = False
            for l in header_lines:
                tokens = l.split()
                if not tokens:
                    continue
                if tokens[0] == 'element' and tokens[1] == 'vertex':
                    num_vertices = int(tokens[2])
                    reading_vertex = True
                elif tokens[0] == 'element' and tokens[1] != 'vertex':
                    reading_vertex = False
                elif reading_vertex and tokens[0] == 'property':
                    p_type = tokens[1]
                    p_name = tokens[2].lower()
                    properties.append((p_name, p_type))

            prop_names = [p[0] for p in properties]
            x_i = prop_names.index('x') if 'x' in prop_names else 0
            y_i = prop_names.index('y') if 'y' in prop_names else 1
            z_i = prop_names.index('z') if 'z' in prop_names else 2
            
            # Find intensity/reflectivity property
            i_i = -1
            for cand in ['intensity', 'scalar_intensity', 'reflectance', 'confidence', 'diffuse_red']:
                if cand in prop_names:
                    i_i = prop_names.index(cand)
                    break

            if not is_binary:
                body = f.read().decode('ascii', errors='ignore').strip().split('\n')
                records = []
                for line in body[:num_vertices]:
                    toks = line.strip().split()
                    if len(toks) >= len(properties):
                        records.append([float(val) for val in toks[:len(properties)]])
                mat = np.array(records, dtype=np.float32)
            else:
                type_dict = {
                    'float': 'f4', 'float32': 'f4', 'double': 'f8', 'float64': 'f8',
                    'int': 'i4', 'int32': 'i4', 'uint': 'u4', 'uint32': 'u4',
                    'short': 'i2', 'int16': 'i2', 'ushort': 'u2', 'uint16': 'u2',
                    'char': 'i1', 'int8': 'i1', 'uchar': 'u1', 'uint8': 'u1'
                }
                np_dtype = np.dtype([(p[0], type_dict.get(p[1], 'f4')) for p in properties])
                rec = np.frombuffer(f.read(), dtype=np_dtype, count=num_vertices)
                mat = np.column_stack([rec[name].astype(np.float32) for name in prop_names])

            pts = mat[:, [x_i, y_i, z_i]]
            intensity = mat[:, i_i] if i_i >= 0 else np.zeros(len(pts), dtype=np.float32)
            if len(intensity) > 0 and np.max(intensity) > 1.0:
                intensity = intensity / max(255.0, float(np.max(intensity)))

            return PointCloudData(
                points=pts,
                intensity=intensity.astype(np.float32),
                metadata={"format": "ply", "path": filepath, "points_count": len(pts)}
            )

    @classmethod
    def save_ply(cls, pcd_data: PointCloudData, filepath: str, mode: str = "binary"):
        """Exports PointCloudData to a valid .ply file."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        pts = pcd_data.points
        N = len(pts)
        intensity = pcd_data.intensity if len(pcd_data.intensity) == N else np.zeros(N, dtype=np.float32)

        header = (
            "ply\n"
            f"format {'binary_little_endian' if mode == 'binary' else 'ascii'} 1.0\n"
            f"element vertex {N}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float intensity\n"
            "end_header\n"
        )
        with open(filepath, 'wb') as f:
            f.write(header.encode('ascii'))
            if mode == 'ascii':
                for i in range(N):
                    f.write(f"{pts[i, 0]:.4f} {pts[i, 1]:.4f} {pts[i, 2]:.4f} {intensity[i]:.4f}\n".encode('ascii'))
            else:
                data = np.column_stack([pts, intensity]).astype(np.float32).tobytes()
                f.write(data)

    @classmethod
    def load_bin(cls, filepath: str) -> PointCloudData:
        """Parses KITTI or automotive binary float32 point cloud file (Nx4 or Nx3)."""
        file_size = os.path.getsize(filepath)
        if file_size % 16 == 0:
            raw = np.fromfile(filepath, dtype=np.float32).reshape(-1, 4)
            pts = raw[:, :3]
            intensity = raw[:, 3]
        elif file_size % 12 == 0:
            pts = np.fromfile(filepath, dtype=np.float32).reshape(-1, 3)
            intensity = np.zeros(len(pts), dtype=np.float32)
        else:
            # Fallback byte read
            raw = np.fromfile(filepath, dtype=np.float32)
            pts = raw[:(len(raw)//3)*3].reshape(-1, 3)
            intensity = np.zeros(len(pts), dtype=np.float32)

        if len(intensity) > 0 and np.max(intensity) > 1.0:
            intensity = intensity / max(255.0, float(np.max(intensity)))

        return PointCloudData(
            points=pts.astype(np.float32),
            intensity=intensity.astype(np.float32),
            metadata={"format": "bin", "path": filepath, "points_count": len(pts)}
        )

    @classmethod
    def save_bin(cls, pcd_data: PointCloudData, filepath: str):
        """Exports PointCloudData to KITTI-style binary float32 [x, y, z, intensity]."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        pts = pcd_data.points
        N = len(pts)
        intensity = pcd_data.intensity if len(pcd_data.intensity) == N else np.zeros(N, dtype=np.float32)
        mat = np.column_stack([pts, intensity]).astype(np.float32)
        mat.tofile(filepath)

    @classmethod
    def load_las(cls, filepath: str) -> PointCloudData:
        """Parses ASPRS LAS binary point format natively (or uses laspy if available)."""
        try:
            import laspy
            las = laspy.read(filepath)
            pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float32)
            intensity = las.intensity.astype(np.float32) / 65535.0 if hasattr(las, 'intensity') else np.zeros(len(pts), dtype=np.float32)
            return PointCloudData(points=pts, intensity=intensity, metadata={"format": "las", "path": filepath})
        except ImportError:
            with open(filepath, 'rb') as f:
                sig = f.read(4)
                if sig != b'LASF':
                    raise ValueError(f"Invalid LAS file signature: {sig}")
                f.seek(96)
                offset_to_points = struct.unpack('<I', f.read(4))[0]
                point_format = struct.unpack('<B', f.read(1))[0]
                record_len = struct.unpack('<H', f.read(2))[0]
                num_points = struct.unpack('<I', f.read(4))[0]

                f.seek(131)
                x_scale, y_scale, z_scale = struct.unpack('<ddd', f.read(24))
                x_offset, y_offset, z_offset = struct.unpack('<ddd', f.read(24))

                f.seek(offset_to_points)
                pts = np.zeros((num_points, 3), dtype=np.float32)
                intensities = np.zeros(num_points, dtype=np.float32)

                for i in range(num_points):
                    raw = f.read(record_len)
                    xi, yi, zi, inten = struct.unpack('<iiiH', raw[:14])
                    pts[i, 0] = xi * x_scale + x_offset
                    pts[i, 1] = yi * y_scale + y_offset
                    pts[i, 2] = zi * z_scale + z_offset
                    intensities[i] = inten / 65535.0

                return PointCloudData(
                    points=pts,
                    intensity=intensities,
                    metadata={"format": "las_native", "path": filepath, "points_count": num_points}
                )

    @classmethod
    def load_npy(cls, filepath: str) -> PointCloudData:
        """Loads .npy or .npz scan files."""
        data = np.load(filepath, allow_pickle=True)
        if isinstance(data, np.ndarray):
            if data.dtype == object and data.size == 1:
                data_dict = data.item()
                return cls.load_dict_or_ros(data_dict)
            pts = data[:, :3].astype(np.float32)
            inten = data[:, 3].astype(np.float32) if data.shape[1] > 3 else np.zeros(len(pts), dtype=np.float32)
            if np.max(inten) > 1.0:
                inten = inten / max(255.0, float(np.max(inten)))
            return PointCloudData(points=pts, intensity=inten, metadata={"format": "npy", "path": filepath})
        elif hasattr(data, 'files'): # .npz
            data_dict = {k: data[k] for k in data.files}
            return cls.load_dict_or_ros(data_dict)
        else:
            raise ValueError(f"Unrecognized NumPy data structure in {filepath}")

    @classmethod
    def load_dict_or_ros(cls, data_dict: Dict[str, Any]) -> PointCloudData:
        """Parses synthetic scan dictionaries or ROS PointCloud2 messages."""
        if "points" in data_dict:
            pts = np.asarray(data_dict["points"], dtype=np.float32)
            inten = np.asarray(data_dict.get("intensity", np.zeros(len(pts))), dtype=np.float32)
            if len(inten) > 0 and np.max(inten) > 1.0:
                inten = inten / max(255.0, float(np.max(inten)))
            rings = np.asarray(data_dict["ring"], dtype=np.int32) if "ring" in data_dict else None
            times = np.asarray(data_dict["timestamp"], dtype=np.float32) if "timestamp" in data_dict else None
            labels = np.asarray(data_dict["labels"], dtype=np.int32) if "labels" in data_dict else None
            meta = {k: v for k, v in data_dict.items() if k not in ["points", "intensity", "ring", "timestamp", "labels"]}
            meta["format"] = "synthetic_dict"
            return PointCloudData(points=pts, intensity=inten, rings=rings, timestamps=times, labels=labels, metadata=meta)
        elif "data" in data_dict and "fields" in data_dict:
            raw_bytes = data_dict["data"]
            point_step = data_dict.get("point_step", 16)
            N = len(raw_bytes) // point_step
            pts = np.zeros((N, 3), dtype=np.float32)
            for i in range(N):
                offset = i * point_step
                pts[i] = struct.unpack_from('<fff', raw_bytes, offset)
            return PointCloudData(points=pts, metadata={"format": "ros_pointcloud2"})
        else:
            raise KeyError("Dictionary must contain either 'points' or ('data' and 'fields').")

class SyntheticStreamGenerator:
    """
    Synchronized 64-Beam Automotive LiDAR Stream Generator:
    Continuously generates dynamic 64-beam LiDAR scans utilizing SyntheticEnvironment
    and LidarSimulator, including terrain, curbs, potholes, dynamic actors, and static poles.
    """

    def __init__(self, seed: int = 42, scene_id: int = 1):
        self.seed = seed
        self.scene_id = scene_id
        self.time = 0.0
        self.ego_speed = 8.0  # m/s
        try:
            from simulation.environment import SyntheticEnvironment
            from simulation.lidar_simulator import LidarSimulator
            self.env = SyntheticEnvironment(scene_id=scene_id, seed=seed)
            self.sim = LidarSimulator(beams=64)
            self._use_sim = True
        except Exception:
            self._use_sim = False
            self.rng = np.random.default_rng(seed)

    def next_frame(self, dt: float = 0.1) -> PointCloudData:
        """Generates next temporal frame of 64-beam point cloud data."""
        self.time += dt

        if self._use_sim:
            self.env.step(dt)
            scan = self.sim.simulate_scan(self.env, timestamp=self.time)
            return PointCloudData(
                points=scan["points"],
                intensity=scan["intensity"],
                rings=scan["ring"],
                timestamps=scan["timestamp"],
                labels=scan["labels"],
                metadata={
                    "format": "synthetic_stream_64beam",
                    "timestamp": self.time,
                    "ego_pose": scan["ego_pose"].tolist(),
                    "gt_boxes": scan.get("gt_boxes", [])
                }
            )

        # Fallback procedural generation
        points = []
        intensities = []
        rings = []
        labels = []

        N_road = 18000
        rx = self.rng.uniform(-25.0, 35.0, N_road)
        ry = self.rng.uniform(-10.0, 10.0, N_road)
        rz = -1.75 + 0.01 * rx + 0.015 * np.sin(0.2 * rx) * np.cos(0.3 * ry)
        road_labels = np.ones(N_road, dtype=np.int32)

        curb_mask = (np.abs(ry) >= 4.3) & (np.abs(ry) <= 4.7)
        rz[curb_mask] += 0.15
        road_labels[curb_mask] = 5

        sidewalk = np.abs(ry) > 4.7
        rz[sidewalk] += 0.15

        potholes = [(10.0, 1.2, 0.9, 0.18), (-12.0, -1.8, 0.7, 0.14)]
        for px, py, pr, pdepth in potholes:
            d = np.hypot(rx - px, ry - py)
            in_p = d < pr
            rz[in_p] -= pdepth * (1.0 - d[in_p] / pr)
            road_labels[in_p] = 6

        road_pts = np.column_stack([rx, ry, rz])
        road_inten = np.full(N_road, 0.35, dtype=np.float32) + self.rng.normal(0, 0.03, N_road)
        road_rings = np.clip(np.floor((rz + 2.0) / 0.1).astype(np.int32), 0, 63)
        points.append(road_pts)
        intensities.append(road_inten)
        rings.append(road_rings)
        labels.append(road_labels)

        # Dynamic Vehicle
        veh_x = 25.0 - (self.time * 12.0) % 70.0
        N_veh = 600
        vx = self.rng.uniform(veh_x - 2.2, veh_x + 2.2, N_veh)
        vy = self.rng.uniform(-3.4, -1.6, N_veh)
        vz = self.rng.uniform(-1.4, 0.1, N_veh)
        points.append(np.column_stack([vx, vy, vz]))
        intensities.append(np.full(N_veh, 0.85, dtype=np.float32))
        rings.append(np.random.randint(15, 45, N_veh))
        labels.append(np.full(N_veh, 3, dtype=np.int32))

        # Pedestrian
        ped_y = 3.0 - (self.time * 1.2) % 6.0
        N_ped = 120
        px = self.rng.uniform(14.7, 15.3, N_ped)
        py = self.rng.uniform(ped_y - 0.3, ped_y + 0.3, N_ped)
        pz = self.rng.uniform(-1.6, 0.1, N_ped)
        points.append(np.column_stack([px, py, pz]))
        intensities.append(np.full(N_ped, 0.65, dtype=np.float32))
        rings.append(np.random.randint(20, 55, N_ped))
        labels.append(np.full(N_ped, 3, dtype=np.int32))

        # Poles
        for pole_x in [-15.0, 5.0, 25.0]:
            N_pole = 80
            px_p = np.full(N_pole, pole_x) + self.rng.normal(0, 0.04, N_pole)
            py_p = np.full(N_pole, 5.5) + self.rng.normal(0, 0.04, N_pole)
            pz_p = np.linspace(-1.6, 2.4, N_pole)
            points.append(np.column_stack([px_p, py_p, pz_p]))
            intensities.append(np.full(N_pole, 0.90, dtype=np.float32))
            rings.append(np.random.randint(10, 60, N_pole))
            labels.append(np.full(N_pole, 2, dtype=np.int32))

        all_pts = np.vstack(points).astype(np.float32)
        all_inten = np.clip(np.concatenate(intensities), 0.0, 1.0).astype(np.float32)
        all_rings = np.concatenate(rings).astype(np.int32)
        all_labels = np.concatenate(labels).astype(np.int32)

        return PointCloudData(
            points=all_pts,
            intensity=all_inten,
            rings=all_rings,
            labels=all_labels,
            metadata={"format": "synthetic_stream", "timestamp": self.time, "ego_speed": self.ego_speed}
        )
