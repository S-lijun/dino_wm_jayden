"""Region-based waypoint sampling and navigation (from DataCollection_loop_test.py)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# Scene shifted +1.5m in x (avoid GS issues near old origin).
# All waypoints shifted -2m in y (aisle centerline at y=-2).
# Obstacle: FIXED at (3.5,-2); training may resample y∈[-2.5,-1.5] on x=3.5.
# start: spawn disk around (0,-2), r=1; then front disk (1.5,-2) r=0.5; then fork.
# left / right: terminal goals beside the bin, disks r=0.5 at (3.5, -0.5) / (3.5, -3.5).
# middle: collision demos for critic (fixed point on the bin).
# back: past the bin on the centerline (straight ahead behind obstacle).
# Fully-safe lateral points for a_good (SF reg when Q(a_nom)<0). Not collect vias.
SAFE_SIDE_WAYPOINTS: tuple[np.ndarray, np.ndarray] = (
    np.array([3.5, -0.5], dtype=np.float64),  # left of bin at (3.5,-2)
    np.array([3.5, -3.5], dtype=np.float64),  # right of bin
)
DEFAULT_TRAJECTORY_REGIONS: dict[str, dict[str, Any]] = {
    "start": {"center": np.array([0.0, -2.0], dtype=np.float64), "r": 1.0},
    "front": {"center": np.array([1.5, -2.0], dtype=np.float64), "r": 0.5},
    "left": {"center": np.array([3.5, -0.5], dtype=np.float64), "r": 0.5},
    "right": {"center": np.array([3.5, -3.5], dtype=np.float64), "r": 0.5},
    "middle": {"mode": "point", "xy": (3.5, -2.0)},
    "back": {"mode": "point", "xy": (5.0, -2.0)},
}

# (0,-2) -> (1.5,-2) -> left|right|middle. No behind-bin back.
# QP critic-only: buffer + train may include middle (collision demos).
# Actor pipelines / test: disable middle via include_middle_pass=False.
DEFAULT_TRAJECTORY_REGION_SEQUENCE: list[str | tuple[str, ...]] = [
    "start",
    "front",
    ("left", "right", "middle"),
]

PASS_SIDE_CYCLE: tuple[str, ...] = ("left", "right", "middle")
PASS_SIDE_TRAIN: tuple[str, ...] = ("left", "right")

# Start–goal path layout (new SAC pipeline only; region layout is unchanged).
DEFAULT_ARENA_BOUNDS: tuple[float, float, float, float] = (-2.0, 5.0, -4.0, 2.0)
DEFAULT_PERP_OFFSET_M: float = 2.5
DEFAULT_MIN_START_GOAL_DIST: float = 4.0


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


def sample_point_in_half_disk(
    region_cfg: dict[str, Any],
    rng: np.random.Generator,
    *,
    side: str,
) -> np.ndarray:
    """Uniform sample in the left (+y) or right (-y) half of a disk region.

    Facing +x: left hemisphere is y >= center_y; right is y <= center_y.
    """
    side = str(side).lower()
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    center = np.asarray(region_cfg["center"], dtype=np.float64).reshape(2)
    for _ in range(256):
        pt = sample_point_in_region(region_cfg, rng)
        if side == "left" and float(pt[1]) >= float(center[1]):
            return pt
        if side == "right" and float(pt[1]) <= float(center[1]):
            return pt
    # Fallback: project a full-disk sample onto the correct half-plane.
    pt = sample_point_in_region(region_cfg, rng)
    if side == "left" and float(pt[1]) < float(center[1]):
        pt = pt.copy()
        pt[1] = float(center[1]) + abs(float(pt[1]) - float(center[1]))
    elif side == "right" and float(pt[1]) > float(center[1]):
        pt = pt.copy()
        pt[1] = float(center[1]) - abs(float(pt[1]) - float(center[1]))
    return pt


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


def _point_in_rect(
    xy: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    margin: float = 0.0,
) -> bool:
    x, y = float(xy[0]), float(xy[1])
    return (
        (x_min + margin) <= x <= (x_max - margin)
        and (y_min + margin) <= y <= (y_max - margin)
    )


def _clamp_to_rect(
    xy: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    margin: float = 0.0,
) -> np.ndarray:
    return np.array(
        [
            float(np.clip(xy[0], x_min + margin, x_max - margin)),
            float(np.clip(xy[1], y_min + margin, y_max - margin)),
        ],
        dtype=np.float64,
    )


def sample_start_goal_perp_path(
    rng: np.random.Generator,
    *,
    x_min: float = DEFAULT_ARENA_BOUNDS[0],
    x_max: float = DEFAULT_ARENA_BOUNDS[1],
    y_min: float = DEFAULT_ARENA_BOUNDS[2],
    y_max: float = DEFAULT_ARENA_BOUNDS[3],
    perp_offset: float = DEFAULT_PERP_OFFSET_M,
    min_start_goal_dist: float = DEFAULT_MIN_START_GOAL_DIST,
    margin: float = 0.5,
    t_lo: float = 0.2,
    t_hi: float = 0.8,
    min_t_gap: float = 0.15,
    max_tries: int = 256,
) -> tuple[np.ndarray, list[str], float]:
    """Sample start, two perpendicular transition waypoints, and goal.

    Transition waypoints sit on the start→goal segment, then are offset
    perpendicularly by ``±perp_offset`` (random independent signs). They are
    ordered so the one closer to the start comes first (no turning back).

    Returns ``(waypoints (4, 2), names, spawn_yaw)``. ``spawn_yaw`` faces
    start → first transition waypoint.
    """
    ix0, ix1 = float(x_min + margin), float(x_max - margin)
    iy0, iy1 = float(y_min + margin), float(y_max - margin)
    if ix1 <= ix0 or iy1 <= iy0:
        raise ValueError("Arena inset is empty; increase bounds or reduce margin.")

    last_err = "failed to sample start-goal path"
    for _ in range(int(max_tries)):
        start = np.array(
            [rng.uniform(ix0, ix1), rng.uniform(iy0, iy1)], dtype=np.float64
        )
        goal = np.array(
            [rng.uniform(ix0, ix1), rng.uniform(iy0, iy1)], dtype=np.float64
        )
        delta = goal - start
        dist = float(np.linalg.norm(delta))
        if dist < float(min_start_goal_dist):
            last_err = f"start-goal dist {dist:.2f} < {min_start_goal_dist}"
            continue
        tangent = delta / dist
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

        t1 = float(rng.uniform(t_lo, t_hi))
        t2 = float(rng.uniform(t_lo, t_hi))
        if abs(t1 - t2) < float(min_t_gap):
            last_err = "transition t too close"
            continue

        trans: list[np.ndarray] = []
        ok = True
        for t in (t1, t2):
            foot = start + t * delta
            sign = -1.0 if rng.random() < 0.5 else 1.0
            pt = foot + sign * float(perp_offset) * normal
            if not _point_in_rect(pt, x_min, x_max, y_min, y_max, margin=0.3):
                pt = foot - sign * float(perp_offset) * normal
            if not _point_in_rect(pt, x_min, x_max, y_min, y_max, margin=0.15):
                pt = _clamp_to_rect(pt, x_min, x_max, y_min, y_max, margin=0.15)
            if not _point_in_rect(pt, x_min, x_max, y_min, y_max, margin=0.05):
                ok = False
                last_err = "transition waypoint outside arena"
                break
            trans.append(pt)
        if not ok:
            continue

        d0 = float(np.linalg.norm(trans[0] - start))
        d1 = float(np.linalg.norm(trans[1] - start))
        if d1 < d0:
            trans[0], trans[1] = trans[1], trans[0]

        heading = trans[0] - start
        if float(np.linalg.norm(heading)) < 1e-6:
            heading = goal - start
        spawn_yaw = float(np.arctan2(heading[1], heading[0]))
        wps = np.stack([start, trans[0], trans[1], goal], axis=0)
        names = ["start", "trans1", "trans2", "goal"]
        return wps, names, spawn_yaw

    raise RuntimeError(f"sample_start_goal_perp_path: {last_err}")


def sample_perp_points_on_segment(
    rng: np.random.Generator,
    start: np.ndarray,
    goal: np.ndarray,
    n: int,
    *,
    perp_offset: float = DEFAULT_PERP_OFFSET_M,
    x_min: float = DEFAULT_ARENA_BOUNDS[0],
    x_max: float = DEFAULT_ARENA_BOUNDS[1],
    y_min: float = DEFAULT_ARENA_BOUNDS[2],
    y_max: float = DEFAULT_ARENA_BOUNDS[3],
    t_lo: float = 0.25,
    t_hi: float = 0.75,
    min_t_gap: float = 0.18,
    min_pairwise: float = 1.2,
    min_end_dist: float = 2.0,
    margin: float = 0.3,
    max_tries: int = 256,
) -> list[np.ndarray]:
    """Sample ``n`` points offset ±perp_offset from the start→goal segment.

    Used for path-layout obstacles (and similar geometry). Empty if ``n<=0``.
    """
    n = int(n)
    if n <= 0:
        return []
    start = np.asarray(start, dtype=np.float64).reshape(2)
    goal = np.asarray(goal, dtype=np.float64).reshape(2)
    delta = goal - start
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        raise ValueError("start and goal are coincident")
    tangent = delta / dist
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)

    last_err = "failed to sample perp points"
    for _ in range(int(max_tries)):
        ts = [float(rng.uniform(t_lo, t_hi)) for _ in range(n)]
        ts.sort()
        if any(abs(ts[i] - ts[i - 1]) < float(min_t_gap) for i in range(1, n)):
            last_err = "t too close"
            continue
        pts: list[np.ndarray] = []
        ok = True
        for t in ts:
            foot = start + t * delta
            sign = -1.0 if rng.random() < 0.5 else 1.0
            pt = foot + sign * float(perp_offset) * normal
            if not _point_in_rect(pt, x_min, x_max, y_min, y_max, margin=margin):
                pt = foot - sign * float(perp_offset) * normal
            if not _point_in_rect(pt, x_min, x_max, y_min, y_max, margin=0.15):
                pt = _clamp_to_rect(pt, x_min, x_max, y_min, y_max, margin=0.15)
            if not _point_in_rect(pt, x_min, x_max, y_min, y_max, margin=0.05):
                ok = False
                last_err = "point outside arena"
                break
            if float(np.linalg.norm(pt - start)) < float(min_end_dist):
                ok = False
                last_err = "too close to start"
                break
            if float(np.linalg.norm(pt - goal)) < float(min_end_dist):
                ok = False
                last_err = "too close to goal"
                break
            pts.append(pt)
        if not ok:
            continue
        too_close = False
        for i in range(n):
            for j in range(i + 1, n):
                if float(np.linalg.norm(pts[i] - pts[j])) < float(min_pairwise):
                    too_close = True
                    last_err = "pairwise too close"
                    break
            if too_close:
                break
        if too_close:
            continue
        return pts
    raise RuntimeError(f"sample_perp_points_on_segment: {last_err}")


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
