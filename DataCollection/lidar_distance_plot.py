from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Robot copy of this folder. Used when that path exists.
ROBOT_BASE_DIR = Path(
    "/home/unitree/Installed_Libraries/unitree_sdk2_python/example/g1/high_level/DataCollection"
)
# Folder that contains this script. Used on Windows, or anywhere the robot path is absent.
SCRIPT_BASE_DIR = Path(__file__).resolve().parent

if ROBOT_BASE_DIR.is_dir():
    BASE_DIR = ROBOT_BASE_DIR
else:
    BASE_DIR = SCRIPT_BASE_DIR

print(f"Using data directory: {BASE_DIR}")

for traj_dir in BASE_DIR.iterdir():
    if not traj_dir.is_dir():
        continue

    csv_path = traj_dir / f"lidar_distance_analysis_{traj_dir.name}.csv"

    if not csv_path.exists():
        continue

    df = pd.read_csv(csv_path)

    distance_col = "filtered_min_distance"

    if distance_col not in df.columns:
        print(f"[SKIP] {traj_dir.name}: no {distance_col} column")
        print("Columns found:", list(df.columns))
        continue

    if "frame" in df.columns:
        x = df["frame"].astype(int)
    else:
        x = range(len(df))

    plt.figure(figsize=(8, 4))
    plt.plot(x, df[distance_col], marker="o")
    plt.xlabel("Step")
    plt.ylabel("Closest distance (m)")
    plt.title(f"Closest LiDAR Distance vs Step\n{traj_dir.name}")
    plt.grid(True)

    output_path = traj_dir / f"lidar_distance_plot_{traj_dir.name}.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] saved {output_path}")

    scan_col = "lidar360_min_distance"
    if scan_col not in df.columns:
        print(f"[SKIP] {traj_dir.name}: no {scan_col} column")
        continue

    plt.figure(figsize=(8, 4))
    plt.plot(x, df[scan_col], marker="o")
    plt.xlabel("Step")
    plt.ylabel("Closest distance (m)")
    plt.title(f"360° LiDAR min distance vs Step\n{traj_dir.name}")
    plt.grid(True)

    scan_path = traj_dir / f"lidar360_distance_plot_{traj_dir.name}.png"
    plt.savefig(scan_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[OK] saved {scan_path}")