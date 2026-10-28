"""Region-based waypoint sampling and navigation (from DataCollection_loop_test.py)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

# Default circular regions on the XY plane (matches DataCollection_loop_test.py).
DEFAULT_TRAJECTORY_REGIONS: dict[str, dict[str, Any]] = {
    "front": {"center": np.array([0.0, 0.0], dtype=np.float64), "r": 0.5},
    "back": {"center": np.array([3.0, 0.0], dtype=np.float64), "r": 0.3},
    "left": {"center": np.array([2.0, 0.6], dtype=np.float64), "r": 0.6},
    "right": {"center": np.array([2.0, -0.6], dtype=np.float64), "r": 0.6},
}

# front -> (left or right) -> back
DEFAULT_TRAJECTORY_REGION_SEQUENCE: list[str | tuple[str, ...]] = [
    "front",
    ("left", "right"),
    "back",
]


def waypoints_to_list(waypoint: np.ndarray) -> list[np.ndarray]:
    """Single (2,) -> one point; (N, 2) -> N points."""
    w = np.asarray(waypoint, dtype=np.float64)
    if w.ndim == 1 and w.size == 2:
        return [w]
    if w.ndim == 2 and w.shape[1] == 2:
        return [w[i] for i in range(w.shape[0])]
    raise ValueError(f"waypoint must be shape (2,) or (N, 2), got {w.shape}")


def sample_point_in_region(
    center: np.ndarray,
    radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniformly sample a point inside a 2D disk."""
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
        cfg = regions[region_name]
        pt = sample_point_in_region(cfg["center"], float(cfg["r"]), rng)
        points.append(pt)
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


# start→goal arena used by the lidar / vision sac_path trainers.
DEFAULT_ARENA_BOUNDS: tuple[float, float, float, float] = (-1.0, 6.0, -4.0, 2.0)
DEFAULT_PERP_OFFSET_M: float = 1.5
DEFAULT_PERP_KEEPOUT_M: float = 0.25
DEFAULT_MIN_START_GOAL_DIST: float = 5.0


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
    return (x_min + margin) <= x <= (x_max - margin) and (y_min + margin) <= y <= (
        y_max - margin
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


def sample_start_goal_in_arena(
    rng: np.random.Generator,
    *,
    x_min: float = DEFAULT_ARENA_BOUNDS[0],
    x_max: float = DEFAULT_ARENA_BOUNDS[1],
    y_min: float = DEFAULT_ARENA_BOUNDS[2],
    y_max: float = DEFAULT_ARENA_BOUNDS[3],
    min_start_goal_dist: float = DEFAULT_MIN_START_GOAL_DIST,
    margin: float = 0.5,
    max_tries: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample start and goal inside the arena inset, with a min separation."""
    ix0, ix1 = float(x_min + margin), float(x_max - margin)
    iy0, iy1 = float(y_min + margin), float(y_max - margin)
    if ix1 <= ix0 or iy1 <= iy0:
        raise ValueError("Arena inset is empty; increase bounds or reduce margin.")
    last_err = "failed to sample start-goal"
    for _ in range(int(max_tries)):
        start = np.array(
            [rng.uniform(ix0, ix1), rng.uniform(iy0, iy1)], dtype=np.float64
        )
        goal = np.array(
            [rng.uniform(ix0, ix1), rng.uniform(iy0, iy1)], dtype=np.float64
        )
        dist = float(np.linalg.norm(goal - start))
        if dist < float(min_start_goal_dist):
            last_err = f"start-goal dist {dist:.2f} < {min_start_goal_dist}"
            continue
        return start, goal
    raise RuntimeError(f"sample_start_goal_in_arena: {last_err}")


def sample_start_goal_path(
    rng: np.random.Generator,
    *,
    x_min: float = DEFAULT_ARENA_BOUNDS[0],
    x_max: float = DEFAULT_ARENA_BOUNDS[1],
    y_min: float = DEFAULT_ARENA_BOUNDS[2],
    y_max: float = DEFAULT_ARENA_BOUNDS[3],
    min_start_goal_dist: float = DEFAULT_MIN_START_GOAL_DIST,
    margin: float = 0.5,
    max_tries: int = 256,
) -> tuple[np.ndarray, list[str], float]:
    """Straight start → goal. ``spawn_yaw`` faces start → goal."""
    start, goal = sample_start_goal_in_arena(
        rng,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        min_start_goal_dist=min_start_goal_dist,
        margin=margin,
        max_tries=max_tries,
    )
    heading = goal - start
    spawn_yaw = float(np.arctan2(heading[1], heading[0]))
    wps = np.stack([start, goal], axis=0)
    return wps, ["start", "goal"], spawn_yaw


def sample_perp_points_on_segment(
    rng: np.random.Generator,
    start: np.ndarray,
    goal: np.ndarray,
    n: int,
    *,
    perp_offset: float = DEFAULT_PERP_OFFSET_M,
    perp_keepout: float = DEFAULT_PERP_KEEPOUT_M,
    side: int | None = None,
    x_min: float = DEFAULT_ARENA_BOUNDS[0],
    x_max: float = DEFAULT_ARENA_BOUNDS[1],
    y_min: float = DEFAULT_ARENA_BOUNDS[2],
    y_max: float = DEFAULT_ARENA_BOUNDS[3],
    t_lo: float = 0.25,
    t_hi: float = 0.75,
    min_t_gap: float = 0.18,
    min_pairwise: float = 1.2,
    min_end_dist: float = 1.0,
    margin: float = 0.3,
    max_tries: int = 256,
) -> list[np.ndarray]:
    """Sample ``n`` points off the start→goal line.

    ``|offset| ~ U[perp_keepout, perp_offset]``. ``side`` is +1 = left of
    heading, -1 = right; ``None`` draws the sign at random per point.
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
    off = abs(float(perp_offset))
    keep = min(abs(float(perp_keepout)), off)
    side_sign = None if side is None else (1.0 if int(side) >= 0 else -1.0)

    def _draw_offsets() -> list[float]:
        mags = [
            float(keep) if off <= keep else float(rng.uniform(keep, off))
            for _ in range(n)
        ]
        if side_sign is None:
            return [m * (1.0 if rng.random() < 0.5 else -1.0) for m in mags]
        return [side_sign * m for m in mags]

    def _place_at(foot: np.ndarray, signed_off: float) -> np.ndarray | None:
        pt = foot + signed_off * normal
        if _point_in_rect(pt, x_min, x_max, y_min, y_max, margin=margin):
            return pt
        if side_sign is None:
            flipped = foot - signed_off * normal
            if _point_in_rect(flipped, x_min, x_max, y_min, y_max, margin=margin):
                return flipped
            if _point_in_rect(flipped, x_min, x_max, y_min, y_max, margin=0.15):
                return _clamp_to_rect(
                    flipped, x_min, x_max, y_min, y_max, margin=0.15
                )
        sgn = float(np.sign(signed_off)) if abs(signed_off) > 1e-9 else 1.0
        for frac in (0.85, 0.7, 0.55, 0.4):
            mag = max(keep, abs(signed_off) * frac)
            cand = foot + sgn * mag * normal
            if _point_in_rect(cand, x_min, x_max, y_min, y_max, margin=0.15):
                return cand
        return None

    for relax in (1.0, 0.7, 0.45, 0.25, 0.0):
        end_min = float(min_end_dist) * relax
        pair_min = float(min_pairwise) * relax
        gap = float(min_t_gap) * relax
        for _ in range(int(max_tries)):
            ts = sorted(float(rng.uniform(t_lo, t_hi)) for _ in range(n))
            if any(ts[i] - ts[i - 1] < gap for i in range(1, n)):
                continue
            pts: list[np.ndarray] = []
            ok = True
            for t, signed_off in zip(ts, _draw_offsets()):
                foot = start + t * delta
                pt = _place_at(foot, signed_off)
                if pt is None:
                    ok = False
                    break
                if float(np.linalg.norm(pt - start)) < end_min:
                    ok = False
                    break
                if float(np.linalg.norm(pt - goal)) < end_min:
                    ok = False
                    break
                pts.append(pt)
            if not ok:
                continue
            if pair_min > 0.0 and any(
                float(np.linalg.norm(pts[i] - pts[j])) < pair_min
                for i in range(n)
                for j in range(i + 1, n)
            ):
                continue
            return pts
    raise RuntimeError("sample_perp_points_on_segment: no placement found")
