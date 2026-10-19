import os
from pathlib import Path

import numpy as np
import open3d as o3d

# ===== CHANGE THIS TO ONE OF YOUR TRAJECTORIES =====
trajectory_dir = Path(
    "/home/unitree/Installed_Libraries/unitree_sdk2_python/example/g1/high_level/DataCollection/20260612_015000"
)

# Load first LiDAR frame
lidar_file = sorted(trajectory_dir.glob("lidar_*.npy"))[0]

print(f"Loading: {lidar_file.name}")

points = np.load(lidar_file)
print("Point cloud shape:", points.shape)

xyz = points[:, :3]

# Create point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

# If no graphical display exists (SSH), export instead
if "DISPLAY" not in os.environ:
    out_file = trajectory_dir / f"{lidar_file.stem}_pointcloud.ply"
    o3d.io.write_point_cloud(str(out_file), pcd)

    print(f"\n[OK] No display detected.")
    print(f"[OK] Saved point cloud to:\n{out_file}")
    print("Copy this .ply file to your computer and open it in CloudCompare or MeshLab.")
else:
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    o3d.visualization.draw_geometries(
        [pcd, axis],
        window_name="LiDAR Point Cloud",
        width=1200,
        height=800,
    )