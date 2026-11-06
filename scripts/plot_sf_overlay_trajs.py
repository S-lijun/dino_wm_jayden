#!/usr/bin/env python3
"""Overlay waypoint-only and the three SF trajectories for each seed-0 trial.

Color distinguishes the controller. A thicker stroke of the same color marks
steps where that safety filter was executing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

REPO = Path(__file__).resolve().parents[1]

CURVES = (
    ("waypoint", "Waypoint only", "0.45", 2),
    ("vision", "Vision only", "#1f77b4", 3),
    ("lidar", "Lidar only", "#d62728", 4),
    ("vision_lidar", "Vision + lidar", "#2ca02c", 5),
)
THIN_LW = 1.5
THICK_LW = 4.0
ARENA = (-1.0, 6.0, -4.0, 2.0)
GOAL_RADIUS = 0.1
DANGER_RADIUS = 1.0


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        xy = np.asarray(z["xy"], dtype=np.float64).reshape(-1, 2)
        use_sf = np.asarray(z["use_sf"], dtype=bool).reshape(-1)
        if use_sf.shape[0] + 1 == xy.shape[0]:
            use_sf = np.concatenate([np.zeros(1, dtype=bool), use_sf])
        if use_sf.shape[0] != xy.shape[0]:
            raise ValueError(f"{path}: use_sf {use_sf.shape} vs xy {xy.shape}")
        return {
            "xy": xy,
            "use_sf": use_sf,
            "waypoints": np.asarray(z["waypoints"], dtype=np.float64).reshape(-1, 2),
            "obstacle_xys": np.asarray(z["obstacle_xys"], dtype=np.float64).reshape(-1, 2),
            "end_reason": str(z["end_reason"]),
        }


def _plot_sf_thick(ax, xy: np.ndarray, use_sf: np.ndarray, color: str, zorder: int) -> None:
    """Thick stroke on segments that arrived under the safety filter."""
    on = np.asarray(use_sf, dtype=bool).copy()
    on[0] = False
    i = 1
    n = on.shape[0]
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < n and on[j]:
            j += 1
        seg = xy[i - 1 : j]
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            color=color,
            linewidth=THICK_LW,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=zorder,
        )
        i = j


def plot_run(curves: dict[str, dict], out_path: Path, run_id: int) -> None:
    ref = curves["waypoint"]
    wps = ref["waypoints"]
    bins = ref["obstacle_xys"]
    for key, data in curves.items():
        if not np.allclose(data["waypoints"], wps):
            raise ValueError(f"run{run_id:02d} {key} waypoints differ")
        if data["obstacle_xys"].shape != bins.shape or not np.allclose(
            data["obstacle_xys"], bins
        ):
            raise ValueError(f"run{run_id:02d} {key} bins differ")

    fig, ax = plt.subplots(figsize=(7.6, 5.8), dpi=150)
    x0, x1, y0, y1 = ARENA
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="0.55",
            linewidth=1.0,
            zorder=1,
        )
    )
    ax.set_xlim(x0 - 0.4, x1 + 0.4)
    ax.set_ylim(y0 - 0.4, y1 + 0.4)

    for key, label, color, z in CURVES:
        xy = curves[key]["xy"]
        ax.plot(
            xy[:, 0],
            xy[:, 1],
            color=color,
            linewidth=THIN_LW,
            solid_capstyle="round",
            zorder=z,
            label=label,
        )
    for key, _label, color, z in CURVES:
        _plot_sf_thick(ax, curves[key]["xy"], curves[key]["use_sf"], color, z + 10)

    ax.scatter(
        wps[0, 0],
        wps[0, 1],
        facecolors="limegreen",
        edgecolors="black",
        linewidths=0.8,
        s=55,
        zorder=30,
        label="start",
    )
    ax.scatter(
        wps[-1, 0],
        wps[-1, 1],
        c="red",
        s=90,
        marker="*",
        zorder=30,
        label="goal",
    )
    ax.add_patch(
        Circle(
            (float(wps[-1, 0]), float(wps[-1, 1])),
            GOAL_RADIUS,
            fill=False,
            edgecolor="red",
            linestyle="--",
            linewidth=1.0,
            zorder=29,
        )
    )
    for i, xy in enumerate(bins):
        ax.scatter(
            float(xy[0]),
            float(xy[1]),
            c="blue",
            s=36,
            zorder=28,
            label="bin" if i == 0 else None,
        )
        ax.add_patch(
            Circle(
                (float(xy[0]), float(xy[1])),
                DANGER_RADIUS,
                fill=False,
                edgecolor="blue",
                linestyle="--",
                linewidth=1.1,
                zorder=27,
                label="bin r=1 m" if i == 0 else None,
            )
        )

    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color="black", linewidth=THICK_LW, solid_capstyle="round"))
    labels.append("SF active")
    n_bin = int(bins.shape[0])
    bin_tag = f"{n_bin} bin" if n_bin == 1 else f"{n_bin} bins"
    ax.set_title(f"run{run_id:02d} | {bin_tag}", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(handles, labels, loc="best", fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vision_dir",
        type=Path,
        default=REPO / "humanoid_test_sac_path_vision/1105_154633",
    )
    parser.add_argument(
        "--lidar_dir",
        type=Path,
        default=REPO / "humanoid_test_sac_path_lidar/1104_134002",
    )
    parser.add_argument(
        "--vision_lidar_dir",
        type=Path,
        default=REPO / "humanoid_test_sac_path_vision_lidar/1103_171300",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=REPO / "humanoid_test_overlay/seed0",
    )
    parser.add_argument("--num_runs", type=int, default=100)
    args = parser.parse_args()

    for run_id in range(int(args.num_runs)):
        tag = f"run{run_id:02d}"
        curves = {
            "waypoint": _load(args.vision_dir / "trajs" / f"{tag}_waypoint_only.npz"),
            "vision": _load(args.vision_dir / "trajs" / f"{tag}_switching.npz"),
            "lidar": _load(args.lidar_dir / "trajs" / f"{tag}_switching.npz"),
            "vision_lidar": _load(args.vision_lidar_dir / "trajs" / f"{tag}_switching.npz"),
        }
        plot_run(curves, args.out_dir / f"{tag}_overlay.png", run_id)
    print(f"[INFO] Wrote {args.num_runs} plots to {args.out_dir}")


if __name__ == "__main__":
    main()
