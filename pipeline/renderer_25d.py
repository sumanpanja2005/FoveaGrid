import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, List, Dict, Any

from .dem_elevation import DEMGrid
from .bev_projection import BEVGrid
from .feature_extraction import DetectedFeatures, ObstacleCluster

def create_false_color_elevation_cmap():
    """
    Creates a customized false-color elevation gradient:
    Cool tones (deep navy blue, royal cyan) for depressions/potholes/ground,
    through emerald green for flat pavement, to warm tones (gold, orange, crimson red)
    for elevated obstacles, vehicles, and poles.
    """
    colors = [
        (0.00, "#081d58"),  # Deep Navy (pothole depression)
        (0.15, "#1d91c0"),  # Electric Cyan (ground road base)
        (0.30, "#41b6c4"),  # Turquoise (sidewalk / road)
        (0.45, "#7fcdbb"),  # Seafoam Green
        (0.60, "#fed976"),  # Gold (low curb / bumper)
        (0.75, "#feb24c"),  # Bright Amber (pedestrian / hood)
        (0.90, "#f03b20"),  # Vibrant Orange (vehicle roof)
        (1.00, "#bd0026")   # Deep Crimson (tall poles, trees, overhead clearance)
    ]
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, hex_val in colors:
        rgb = [int(hex_val[i:i+2], 16) / 255.0 for i in (1, 3, 5)]
        cdict['red'].append((pos, rgb[0], rgb[0]))
        cdict['green'].append((pos, rgb[1], rgb[1]))
        cdict['blue'].append((pos, rgb[2], rgb[2]))

    return LinearSegmentedColormap('ElevationFalseColor', cdict)

class ElevationRenderer25D:
    """
    Renders 2.5D Digital Elevation Models and BEV representations:
    Isometric 3D false-color surface visualization with 3D obstacle bounding boxes.
    """

    def __init__(self, cmap: Optional[LinearSegmentedColormap] = None):
        self.cmap = cmap or create_false_color_elevation_cmap()

    def render_isometric_dem(
        self,
        dem: DEMGrid,
        features: Optional[DetectedFeatures] = None,
        save_path: Optional[str] = None,
        show_interactive: bool = False,
        elev: float = 32.0,
        azim: float = -45.0,
        stride: int = 2
    ):
        """
        Renders isometric 3D surface plot with false-color elevation gradient and obstacle overlays.
        """
        fig = plt.figure(figsize=(14, 10), facecolor="#0a0e17")
        ax = fig.add_subplot(111, projection='3d', facecolor="#0a0e17")

        # Downsample grid for responsive rendering if dense
        x_mesh = dem.mesh_x[::stride, ::stride]
        y_mesh = dem.mesh_y[::stride, ::stride]
        z_mesh = dem.elevation[::stride, ::stride]

        z_min = float(np.nanmin(z_mesh))
        z_max = max(z_min + 1.0, float(np.nanmax(z_mesh)))
        norm = Normalize(vmin=z_min, vmax=z_max)
        colors = self.cmap(norm(z_mesh))

        # 1. Render 2.5D Isometric Surface
        surf = ax.plot_surface(
            x_mesh, y_mesh, z_mesh,
            facecolors=colors,
            rstride=1, cstride=1,
            linewidth=0.1,
            edgecolor=(0.1, 0.2, 0.3, 0.2),
            antialiased=True,
            shade=True
        )

        # 2. Draw 3D Bounding Boxes for Detected Obstacles
        if features is not None:
            # Dynamic objects (Vehicles / Pedestrians)
            for obs in features.dynamic_obstacles:
                col = "#e74c3c" if obs.category == "vehicle" else "#e67e22"
                self._draw_3d_box(ax, obs.position, obs.size, color=col, label=obs.category)

            # Roadside Infrastructure (Poles / Trees)
            for infra in features.roadside_infrastructure:
                col = "#f1c40f" if infra.category == "pole" else "#27ae60"
                self._draw_3d_box(ax, infra.position, infra.size, color=col, label=infra.category)

            # Potholes markers
            for ph in features.potholes:
                cx, cy, cz = ph["center"]
                ax.scatter(cx, cy, cz - 0.1, color="#e056fd", s=80, marker="v", depthshade=False)
                ax.text(cx, cy, cz + 0.3, f"Pothole\n-{ph['max_depth']:.2f}m", color="#e056fd", fontsize=8, weight='bold')

            # Curbs
            if len(features.curb_points) > 0:
                curb_pts = features.curb_points
                sub = np.random.choice(len(curb_pts), size=min(400, len(curb_pts)), replace=False)
                ax.scatter(curb_pts[sub, 0], curb_pts[sub, 1], curb_pts[sub, 2] + 0.05, color="#00d2d3", s=6, alpha=0.7)

        # 3. View Angle & Aesthetics
        ax.view_init(elev=elev, azim=azim)
        ax.set_title("FoveaGrid: 2.5D Isometric Elevation Model (False-Color Gradient)", color="#ffffff", fontsize=14, weight='bold', pad=15)
        ax.set_xlabel("X (m) — Forward Heading", color="#a4b0be", labelpad=10)
        ax.set_ylabel("Y (m) — Lateral", color="#a4b0be", labelpad=10)
        ax.set_zlabel("Elevation Z (m)", color="#a4b0be", labelpad=10)

        # Dark theme axes styling
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#1e272e')
        ax.yaxis.pane.set_edgecolor('#1e272e')
        ax.zaxis.pane.set_edgecolor('#1e272e')
        ax.tick_params(colors='#a4b0be')

        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.55, aspect=15, pad=0.08)
        cbar.set_label("Elevation Height Z (m)", color="#ffffff")
        cbar.ax.yaxis.set_tick_params(color='#ffffff')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#ffffff')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"[Renderer] Saved 2.5D Isometric Elevation view to: {save_path}")

        if show_interactive:
            plt.show()
        else:
            plt.close(fig)

    def render_comprehensive_dashboard(
        self,
        dem: DEMGrid,
        bev: BEVGrid,
        features: DetectedFeatures,
        save_path: Optional[str] = None,
        show_interactive: bool = False
    ):
        """
        Renders a 4-panel dashboard containing:
        1. 2.5D Isometric Elevation Model with 3D Bounding Boxes
        2. Top-down 2D BEV Occupancy Grid & Hazard Overlay
        3. 2D Height Span ΔZ (Obstacle Clearance)
        4. Mean LiDAR Intensity / Reflectivity Map
        """
        fig = plt.figure(figsize=(18, 12), facecolor="#0a0e17")

        # 1. Isometric 3D DEM (Top-Left)
        ax1 = fig.add_subplot(221, projection='3d', facecolor="#0a0e17")
        stride = 2
        xm = dem.mesh_x[::stride, ::stride]
        ym = dem.mesh_y[::stride, ::stride]
        zm = dem.elevation[::stride, ::stride]
        norm = Normalize(vmin=float(np.nanmin(zm)), vmax=max(float(np.nanmin(zm)) + 1.0, float(np.nanmax(zm))))
        ax1.plot_surface(xm, ym, zm, facecolors=self.cmap(norm(zm)), rstride=1, cstride=1, linewidth=0.1, shade=True)
        ax1.view_init(elev=32, azim=-45)
        ax1.set_title("1. Isometric 2.5D Elevation Model", color="#ffffff", weight='bold', fontsize=12)
        ax1.tick_params(colors='#a4b0be')
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False

        # 2. 2D BEV Occupancy Grid & Feature Detections (Top-Right)
        ax2 = fig.add_subplot(222, facecolor="#0a0e17")
        occ_img = ax2.imshow(
            bev.occupancy, origin='lower',
            extent=[bev.bounds[0], bev.bounds[1], bev.bounds[2], bev.bounds[3]],
            cmap='gray', vmin=0, vmax=1
        )
        # Overlay curbs
        if len(features.curb_points) > 0:
            ax2.scatter(features.curb_points[:, 0], features.curb_points[:, 1], c="#00d2d3", s=2, label="Curb / Boundary")
        # Overlay potholes
        for ph in features.potholes:
            circle = plt.Circle((ph["center"][0], ph["center"][1]), ph["radius"], color="#e056fd", fill=False, linewidth=2)
            ax2.add_patch(circle)
        # Overlay obstacles
        for obs in features.dynamic_obstacles:
            cx, cy = obs.position[0], obs.position[1]
            dx, dy = obs.size[0], obs.size[1]
            rect = plt.Rectangle((cx - dx/2, cy - dy/2), dx, dy, fill=False, edgecolor="#e74c3c" if obs.category=="vehicle" else "#e67e22", linewidth=1.8)
            ax2.add_patch(rect)
            ax2.text(cx - dx/2, cy + dy/2 + 0.4, obs.category, color="#ffffff", fontsize=8, weight='bold')

        for infra in features.roadside_infrastructure:
            cx, cy = infra.position[0], infra.position[1]
            ax2.scatter(cx, cy, c="#f1c40f", s=40, marker="^")
            ax2.text(cx + 0.5, cy, infra.category, color="#f1c40f", fontsize=8)

        ax2.set_title("2. 2D BEV Occupancy & Detected Features", color="#ffffff", weight='bold', fontsize=12)
        ax2.set_xlabel("X (m)", color="#a4b0be")
        ax2.set_ylabel("Y (m)", color="#a4b0be")
        ax2.tick_params(colors='#a4b0be')
        ax2.legend(loc="upper right", facecolor="#1e272e", edgecolor="#a4b0be", labelcolor="#ffffff")

        # 3. 2D Height Span Delta-Z (Bottom-Left)
        ax3 = fig.add_subplot(223, facecolor="#0a0e17")
        im_dz = ax3.imshow(
            bev.z_delta, origin='lower',
            extent=[bev.bounds[0], bev.bounds[1], bev.bounds[2], bev.bounds[3]],
            cmap='plasma'
        )
        ax3.set_title("3. Obstacle Elevation Span (Delta-Z = Zmax - Zmin)", color="#ffffff", weight='bold', fontsize=12)
        ax3.set_xlabel("X (m)", color="#a4b0be")
        ax3.set_ylabel("Y (m)", color="#a4b0be")
        ax3.tick_params(colors='#a4b0be')
        cb3 = fig.colorbar(im_dz, ax=ax3, fraction=0.046, pad=0.04)
        cb3.set_label("Span Delta-Z (m)", color="#ffffff")
        cb3.ax.yaxis.set_tick_params(color='#ffffff')
        plt.setp(plt.getp(cb3.ax.axes, 'yticklabels'), color='#ffffff')

        # 4. Mean LiDAR Intensity / Reflectivity (Bottom-Right)
        ax4 = fig.add_subplot(224, facecolor="#0a0e17")
        im_int = ax4.imshow(
            bev.intensity, origin='lower',
            extent=[bev.bounds[0], bev.bounds[1], bev.bounds[2], bev.bounds[3]],
            cmap='cividis'
        )
        ax4.set_title("4. LiDAR Reflectivity / Mean Intensity", color="#ffffff", weight='bold', fontsize=12)
        ax4.set_xlabel("X (m)", color="#a4b0be")
        ax4.set_ylabel("Y (m)", color="#a4b0be")
        ax4.tick_params(colors='#a4b0be')
        cb4 = fig.colorbar(im_int, ax=ax4, fraction=0.046, pad=0.04)
        cb4.set_label("Intensity [0..1]", color="#ffffff")
        cb4.ax.yaxis.set_tick_params(color='#ffffff')
        plt.setp(plt.getp(cb4.ax.axes, 'yticklabels'), color='#ffffff')

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"[Renderer] Saved Comprehensive 4-Panel Dashboard to: {save_path}")

        if show_interactive:
            plt.show()
        else:
            plt.close(fig)

    @staticmethod
    def _draw_3d_box(ax, pos, size, color="#e74c3c", label=""):
        """Draws wireframe 3D bounding box."""
        cx, cy, cz = pos
        dx, dy, dz = size
        x_min, x_max = cx - dx/2, cx + dx/2
        y_min, y_max = cy - dy/2, cy + dy/2
        z_min, z_max = cz - dz/2, cz + dz/2

        corners = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max]
        ])

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # Top
            (0, 4), (1, 5), (2, 6), (3, 7)   # Verticals
        ]

        for p1, p2 in edges:
            ax.plot(
                [corners[p1, 0], corners[p2, 0]],
                [corners[p1, 1], corners[p2, 1]],
                [corners[p1, 2], corners[p2, 2]],
                color=color, linewidth=1.5
            )

        if label:
            ax.text(cx, cy, z_max + 0.2, label, color=color, fontsize=8, weight='bold')
