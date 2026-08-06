"""Region-based waypoint sampling and navigation (from DataCollection_loop_test.py)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# Scene shifted +1.5m in x (avoid GS issues near old origin).
# Obstacle: FIXED at (3.5,0) during buffer; training resamples y∈[-0.5,0.5] on x=3.5.
# start: spawn disk around (0,0), r=1; then front disk (1.5,0) r=0.5; then fork.
# left / right: terminal goals beside the bin, disks r=0.5 at (3.5, ±1).
# middle: collision demos for critic (fixed point on the bin).
# back: past the bin on the centerline (straight ahead behind obstacle).
DEFAULT_TRAJECTORY_REGIONS: dict[str, dict[str, Any]] = {
    "start": {"center": np.array([0.0, 0.0], dtype=np.float64), "r": 1.0},
    "front": {"center": np.array([1.5, 0.0], dtype=np.float64), "r": 0.5},
    "left": {"center": np.array([3.5, 1.0], dtype=np.float64), "r": 0.5},
    "right": {"center": np.array([3.5, -1.0], dtype=np.float64), "r": 0.5},
    "middle": {"mode": "point", "xy": (3.5, 0.0)},
    "back": {"mode": "point", "xy": (5.0, 0.0)},
}

# (0,0) -> (1.5,0) -> left|right|middle. No behind-bin back.
# QP critic-only: buffer + train may include middle (collision demos).
# Actor pipelines / test: disable middle via include_middle_pass=False.
DEFAULT_TRAJECTORY_REGION_SEQUENCE: list[str | tuple[str, ...]] = [
    "start",
    "front",
    ("left", "right", "middle"),
]

PASS_SIDE_CYCLE: tuple[str, ...] = ("left", "right", "middle")
PASS_SIDE_TRAIN: tuple[str, ...] = ("left", "right")


def waypoints_to_list(waypoint: np.ndarray) -> list[np.ndarray]:
    """Single (2,) -> one point; (N, 2) -> N points."""
    w = np.asarray(waypoint, dtype=np.float64)
    if w.ndim == 1 and w.size == 2:
        return [w]
    if w.ndim == 2 and w.shape[1] == 2:
        return [w[i] for i in range(w.shape[0])]
    raise ValueError(f"waypoint must be shape (2,) or (N, 2), got {w.shape}")


def sample_point_in_region(
    region_cfg: dict[str, Any],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a point from a region cfg (fixed point, disk, or vertical line)."""
    mode = str(region_cfg.get("mode", "disk"))
    if mode == "point":
        xy = region_cfg["xy"]
        return np.array([float(xy[0]), float(xy[1])], dtype=np.float64)
    if mode == "line":
        x = float(region_cfg["x"])
        y = float(rng.uniform(float(region_cfg["y_min"]), float(region_cfg["y_max"])))
        return np.array([x, y], dtype=np.float64)
    # Default: uniform disk
    center = np.asarray(region_cfg["center"], dtype=np.float64)
    radius = float(region_cfg["r"])
    theta = rng.uniform(0.0, 2.0 * np.pi)
    rr = radius * np.sqrt(rng.uniform(0.0, 1.0))
    return center + rr * np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)


def generate_random_waypoint_sequence(
    rng: np.random.Generator,
    *,
    trajectory_regions: dict[str, dict[str, Any]] | None = None,
    trajectory_region_sequence: Sequence[str | tuple[str, ...]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Sample one waypoint per entry in ``trajectory_region_sequence``."""
    regions = DEFAULT_TRAJECTORY_REGIONS if trajectory_regions is None else trajectory_regions
    sequence = (
        DEFAULT_TRAJECTORY_REGION_SEQUENCE
        if trajectory_region_sequence is None
        else trajectory_region_sequence
    )

    points: list[np.ndarray] = []
    log_names: list[str] = []

    for entry in sequence:
        if isinstance(entry, str):
            region_name = entry
        elif isinstance(entry, tuple) and len(entry) > 0 and all(isinstance(x, str) for x in entry):
            region_name = str(rng.choice(entry))
        else:
            raise TypeError(
                "trajectory_region_sequence entries must be str or tuple[str, ...], "
                f"got {type(entry).__name__}: {entry!r}"
            )

        if region_name not in regions:
            raise KeyError(
                f"Region {region_name!r} not in trajectory_regions. Keys: {list(regions.keys())}"
            )
        points.append(sample_point_in_region(regions[region_name], rng))
        log_names.append(region_name)

    return np.stack(points, axis=0), log_names


def quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class WaypointNavController:
    """Shortest-yaw waypoint tracker (ported from DataCollection_loop_test.py)."""

    def __init__(
        self,
        *,
        max_speed: float = 0.5,
        stop_thresh: float = 0.1,
        k_yaw: float = 1.0,
        max_yaw_rate: float = 1.0,
        threshold_deg: float = 55.0,
        dead_zone_deg: float = 30.0,
        vel_smooth_alpha: float = 1.0,
    ):
        self.max_speed = max_speed
        self.stop_thresh = stop_thresh
        self.k_yaw = k_yaw
        self.max_yaw_rate = max_yaw_rate
        self.threshold_deg = threshold_deg
        self.dead_zone_deg = dead_zone_deg
        self.vel_smooth_alpha = vel_smooth_alpha
        self.reset()

    def reset(self) -> None:
        self.prev_yaw_rate = 0.0
        self.prev_yaw = 0.0
        self.prev_vx = 0.0
        self.prev_vy = 0.0

    def compute_command(
        self,
        base_pos: np.ndarray,
        base_quat: np.ndarray,
        target_xy: np.ndarray,
    ) -> np.ndarray:
        yaw = quat_to_yaw(base_quat)
        yaw = float(np.unwrap([self.prev_yaw, yaw])[1])
        self.prev_yaw = yaw

        dx = float(target_xy[0] - base_pos[0])
        dy = float(target_xy[1] - base_pos[1])
        dist = np.hypot(dx, dy)
        if dist < self.stop_thresh:
            return np.zeros(3, dtype=np.float32)

        local_dx = np.cos(yaw) * dx + np.sin(yaw) * dy
        local_dy = -np.sin(yaw) * dx + np.cos(yaw) * dy
        direction_local = np.array([local_dx, local_dy], dtype=np.float64)
        direction_local /= np.linalg.norm(direction_local) + 1e-8

        vx_local = self.max_speed * direction_local[0]
        vy_local = self.max_speed * direction_local[1]
        theta_v = np.arctan2(vy_local, vx_local)
        theta_v = (theta_v + np.pi) % (2 * np.pi) - np.pi
        theta_deg = np.degrees(theta_v)
        dead_zone_start = 180.0 - self.dead_zone_deg

        if -self.threshold_deg <= theta_deg <= self.threshold_deg:
            yaw_rate_to_use = self.k_yaw * theta_v
            vx_cmd = vx_local
            vy_cmd = vy_local
            yaw_smooth = 0.1
        else:
            vx_cmd = 0.1
            vy_cmd = 0.0
            yaw_smooth = 1.0
            if abs(theta_deg) >= dead_zone_start:
                yaw_rate_to_use = self.max_yaw_rate
            else:
                yaw_rate_to_use = np.clip(
                    self.k_yaw * theta_v, -self.max_yaw_rate, self.max_yaw_rate
                )

        yaw_rate = (1.0 - yaw_smooth) * self.prev_yaw_rate + yaw_smooth * yaw_rate_to_use
        self.prev_yaw_rate = float(yaw_rate)

        alpha = self.vel_smooth_alpha
        vx_cmd = (1.0 - alpha) * self.prev_vx + alpha * vx_cmd
        vy_cmd = (1.0 - alpha) * self.prev_vy + alpha * vy_cmd
        self.prev_vx = float(vx_cmd)
        self.prev_vy = float(vy_cmd)

        return np.array([vx_cmd, vy_cmd, yaw_rate], dtype=np.float32)
