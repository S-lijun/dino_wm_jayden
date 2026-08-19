"""Gymnasium env that encodes Isaac G1 observations with a DINO world model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from PIL import Image

from env.isaac.isaac_g1_wrapper import IsaacG1Wrapper
from env.isaac.waypoint_utils import (
    DEFAULT_TRAJECTORY_REGION_SEQUENCE,
    SAFE_SIDE_WAYPOINTS,
    WaypointNavController,
)

# Match datasets/humanoid_dset.py and collect_humanoid_dataset.py
DEFAULT_VISUAL_FPS = 15.0
# 1500 sim steps @ dt=0.005 ≈ 7.5s — too short to finish front→left/right on foot.
# ~8000 ≈ 40s wall-clock sim time at 0.5 m/s with turns.
DEFAULT_MAX_EPISODE_SIM_STEPS = 8000
DEFAULT_WANDB_VIDEO_EVERY = 1
DEFAULT_WANDB_VIDEO_SIZE = (120, 160)  # (H, W) for uploaded rollouts


class LatentHumanoidEnv(gym.Env):
    """Latent-space G1 env for PyHJ avoid-DDPG safety-filter training.

    Time alignment (same as offline WM data):
    - Physics / velocity commands run at sim rate (~200 Hz).
    - One Gym/HJ step = hold the same action until the next 15 fps visual
      sample time, then encode visual+proprio once (1:1 with WM timeline).

    Episode ends (not based on safety cost ``h_s``):
    - all waypoints reached (goal = left or right of the bin; no behind-bin goal)
    - stuck contact on a non-ankle_roll link for ``stuck_contact_steps``
    - sim-step count hits ``max_episode_steps`` (default 8000 control steps)
    - optional soft Y corridor: |y - y_center| > y_bound when y_bound > 0 (default disabled).
      Truncates only — does **not** change ``h_s``.
    - soft X far wall: x >= x_bound_max (default 4.5). Same truncate-only behavior.

    Continuous LiDAR margin ``h_s = lidar_min_distance - 1.0`` (m) from the
    forward LiDAR cone (aisle default ±90°; path pipeline ±60°)
    (<0 unsafe, >0 safe); no in-cone hit → ``h_s = 2.0``.
    Does **not** end the episode.
    Waypoints: start disk (0,-2) r=1 → front (1.5,-2) r=0.5 → left|right r=0.5
    (or middle point); bin at (3.5,-2). Spawn = sampled start (buffer and train).
    """

    metadata = {"render_modes": []}

    # Fixed info schema for PyHJ Batch (reset/step must use the same keys).
    # 0=ongoing, 1=all_waypoints, 2=stuck, 3=max_steps, 4=out_of_bounds
    _END_REASON_CODE = {
        None: 0,
        "all_waypoints": 1,
        "stuck": 2,
        "max_steps": 3,
        "out_of_bounds": 4,
    }

    def __init__(
        self,
        args,
        wm,
        device: str,
        args_cli,
        with_proprio: bool = False,
        latent_h: bool = False,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_SIM_STEPS,
        visual_fps: float = DEFAULT_VISUAL_FPS,
        wandb_video_every: int | None = None,
        wandb_video_size: tuple[int, int] = DEFAULT_WANDB_VIDEO_SIZE,
    ):
        super().__init__()
        self.args = args
        self.device = torch.device(device)
        self.with_proprio = with_proprio
        self.latent_h = latent_h
        self.wm = wm
        self.wm.eval()
        self.max_episode_sim_steps = int(
            getattr(args, "max_episode_steps", max_episode_steps)
        )
        self.visual_fps = float(getattr(args, "visual_fps", visual_fps))
        self.visual_period_s = 1.0 / self.visual_fps

        if wandb_video_every is None:
            wandb_video_every = int(
                getattr(args, "wandb_video_every", DEFAULT_WANDB_VIDEO_EVERY)
            )
        # 0 / negative disables rollout video uploads.
        self.wandb_video_every = int(wandb_video_every)
        self.wandb_video_size = tuple(wandb_video_size)
        self._finished_episodes = 0
        self._record_this_episode = False
        self._episode_frames: list[np.ndarray] = []
        # Set by trainer via DummyVectorEnv.set_env_attr after policy is built.
        self.policy_for_log = None
        # Shared dict with trainer: {"env_step": int, "update": int}
        self.log_state = None
        # Per-step wandb scalars (safety/l, safety/hj). Disable during buffer
        # warm-up — wandb.log can stall Isaac's first RTX step when the service
        # is wedged after a killed run.
        self.log_rollout_to_wandb = True
        # Print timing for the first few visual steps (diagnose "stuck" collects).
        self.debug_step_timing = False
        self._debug_steps_left = 0
        # WM predictor history (visual frames + executed actions) for z_{t+1} look-ahead.
        self._obs_hist: list[dict[str, np.ndarray]] = []
        self._act_hist: list[np.ndarray] = []
        # Formal-train switch episodes: 2D traj PNG (off by default / original pipeline).
        self.record_switch_traj = False
        self.switch_traj_dir = ""
        self.failure_set_radius = 1.5
        self._record_switch_traj_ep = False
        self._switch_traj_xy: list[np.ndarray] = []
        self._switch_traj_use_sf: list[bool] = []

        self._episode_sim_step = 0
        self._episode_visual_step = 0
        self._next_visual_time_s = 0.0

        # enable_cameras is handled by AppLauncher / visual_mode on args_cli,
        # not as an IsaacG1Wrapper kwarg.
        self.wrapper = IsaacG1Wrapper(args_cli)
        self._apply_optional_pipeline_args(args)
        self.sim_dt = float(self.wrapper.sim_dt)
        # Same region-nav controller used in DataCollection_loop_test.
        self.waypoint_nav = WaypointNavController(
            max_speed=float(getattr(self.wrapper, "max_speed", 0.5)),
            stop_thresh=float(self.wrapper.waypoint_stop_thresh),
        )
        # Separate controller so a_good does not share smoothing state with a_nom.
        self._safe_side_nav = WaypointNavController(
            max_speed=float(getattr(self.wrapper, "max_speed", 0.5)),
            stop_thresh=float(self.wrapper.waypoint_stop_thresh),
        )
        self.safe_side_waypoints = tuple(
            np.asarray(p, dtype=np.float64).reshape(2).copy()
            for p in SAFE_SIDE_WAYPOINTS
        )

        if latent_h:
            raise NotImplementedError("FailureClassifier latent_h is not wired for Isaac G1 yet.")

        use_path = (
            str(getattr(self.wrapper, "waypoint_layout", "regions")) == "start_goal_perp"
            or bool(getattr(self.wrapper, "path_obstacle_layout", False))
        )
        reset_info = self.wrapper.reset_scene(
            seed=None if use_path else getattr(args, "seed", None)
        )
        self._reset_timers()
        self.waypoint_nav.reset()
        self._safe_side_nav.reset()
        # Constructor warm-up reset is not a training episode; do not record.
        self._record_this_episode = False
        self._episode_frames = []
        obs = self.wrapper.get_raw_obs()
        z = self.encode(obs)
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=z.shape, dtype=np.float32
        )
        yaw_limit = float(getattr(self.wrapper, "action_high")[2])
        self.action_space = Box(
            low=np.array([0.0, -0.5, -yaw_limit], dtype=np.float32),
            high=np.array([0.8, 0.5, yaw_limit], dtype=np.float32),
            dtype=np.float32,
        )
        self.wrapper.action_low = np.asarray(self.action_space.low, dtype=np.float32)
        self.wrapper.action_high = np.asarray(self.action_space.high, dtype=np.float32)
        self._reset_wm_history(obs)
        approx_substeps = max(1, int(round(self.visual_period_s / self.sim_dt)))
        print(
            f"[LatentHumanoidEnv] latent shape: {z.shape}, "
            f"sim_dt={self.sim_dt:.6f} (~{1.0 / self.sim_dt:.1f} Hz), "
            f"visual_fps={self.visual_fps}, ~{approx_substeps} sim steps / HJ step, "
            f"max_episode_sim_steps={self.max_episode_sim_steps}, "
            f"y_bound={'disabled' if self.wrapper.y_bound <= 0 else f'{self.wrapper.y_center}±{self.wrapper.y_bound}'}, "
            f"x_bound_max={self.wrapper.x_bound_max}, "
            f"arena={'on ' + self._arena_str() if self.wrapper.use_arena_bounds else 'off'}, "
            f"h_s=d_min-{self.wrapper.lidar_distance_threshold:g}"
            f"{('+contact→' + str(self.wrapper.contact_hs) if self.wrapper.include_contact_in_hs else '')}, "
            f"lidar_cone=±{float(getattr(self.wrapper, 'lidar_h_half_fov_deg', 90.0)):g}°, "
            f"yaw=[{self.action_space.low[2]:g},{self.action_space.high[2]:g}] "
            f"(OOB → truncate only, no h_s penalty), "
            f"wm_num_hist={self._wm_num_hist()}, "
            f"wandb_video_every={self.wandb_video_every}, reset: {reset_info}"
        )
        def _region_summary(v: dict) -> dict:
            mode = v.get("mode", "disk")
            if mode == "line":
                return {"mode": "line", "x": v.get("x"), "y": [v.get("y_min"), v.get("y_max")]}
            if mode == "point":
                return {"mode": "point", "xy": list(v.get("xy", (0.0, 0.0)))}
            return {"center": np.asarray(v["center"]).tolist(), "r": v["r"]}

        layout = str(getattr(self.wrapper, "waypoint_layout", "regions"))
        if layout == "start_goal_perp":
            seq_desc = "start → trans1 → trans2 → goal (resampled XY each episode)"
        else:
            seq_desc = (
                f"sequence={DEFAULT_TRAJECTORY_REGION_SEQUENCE} "
                "(goal=left|right|middle; no back); "
                "pass-side: include_middle_pass→left|right|middle "
                "(QP critic train/buffer), else left|right (actor/test)"
            )
        print(
            "[LatentHumanoidEnv] trajectory regions "
            f"{ {k: _region_summary(v) for k, v in self.wrapper.trajectory_regions.items()} }; "
            f"{seq_desc}; "
            f"obstacles: layout={layout} "
            f"max_n={getattr(self.wrapper, '_max_n_obstacles', 1)} "
            f"path_obs={bool(getattr(self.wrapper, 'path_obstacle_layout', False))}"
        )

    @property
    def randomize_obstacle(self) -> bool:
        return bool(self.wrapper.randomize_obstacle)

    @randomize_obstacle.setter
    def randomize_obstacle(self, value: bool) -> None:
        self.wrapper.randomize_obstacle = bool(value)

    @property
    def include_middle_pass(self) -> bool:
        return bool(self.wrapper.include_middle_pass)

    @include_middle_pass.setter
    def include_middle_pass(self, value: bool) -> None:
        self.wrapper.include_middle_pass = bool(value)

    @property
    def force_pass_side(self) -> str | None:
        return getattr(self.wrapper, "force_pass_side", None)

    @force_pass_side.setter
    def force_pass_side(self, value: str | None) -> None:
        """Force pass-side waypoint to this region name, or None to restore cycle."""
        if value is None or value == "" or str(value).lower() in ("none", "null"):
            self.wrapper.force_pass_side = None
        else:
            self.wrapper.force_pass_side = str(value)

    @property
    def advance_passed_terminal(self) -> bool:
        return bool(getattr(self.wrapper, "advance_passed_terminal", False))

    @advance_passed_terminal.setter
    def advance_passed_terminal(self, value: bool) -> None:
        """If True, geometrically passing the last waypoint completes it."""
        self.wrapper.advance_passed_terminal = bool(value)

    @property
    def spawn_hemisphere_pass(self) -> bool:
        return bool(getattr(self.wrapper, "spawn_hemisphere_pass", False))

    @spawn_hemisphere_pass.setter
    def spawn_hemisphere_pass(self, value: bool) -> None:
        """Couple spawn half-disk with matching left/right pass (formal train)."""
        self.wrapper.spawn_hemisphere_pass = bool(value)

    @property
    def start_region(self) -> dict:
        return self.wrapper.trajectory_regions["start"]

    @start_region.setter
    def start_region(self, value: dict) -> None:
        """Override spawn disk, e.g. train: center=(1.5,-2), r=1."""
        self.set_region("start", value)

    @property
    def front_region(self) -> dict:
        return self.wrapper.trajectory_regions["front"]

    @front_region.setter
    def front_region(self, value: dict) -> None:
        """Override front waypoint disk (buffer: start → front → pass-side)."""
        self.set_region("front", value)

    @property
    def trajectory_region_sequence(self):
        return self.wrapper.trajectory_region_sequence

    @trajectory_region_sequence.setter
    def trajectory_region_sequence(self, value) -> None:
        """Override waypoint region order, e.g. ['start', ('left','right')]."""
        self.wrapper.trajectory_region_sequence = list(value)

    def set_region(self, name: str, value: dict) -> None:
        """Set a named trajectory region (disk or point)."""
        cfg = dict(value)
        if "center" in cfg:
            cfg["center"] = np.asarray(cfg["center"], dtype=np.float64).copy()
        if "xy" in cfg:
            xy = cfg["xy"]
            cfg["xy"] = (float(xy[0]), float(xy[1]))
        if "mode" not in cfg and "xy" not in cfg:
            cfg.pop("mode", None)  # disk sampling is the default
        self.wrapper.trajectory_regions[str(name)] = cfg

    @property
    def left_region(self) -> dict:
        return self.wrapper.trajectory_regions["left"]

    @left_region.setter
    def left_region(self, value: dict) -> None:
        self.set_region("left", value)

    @property
    def right_region(self) -> dict:
        return self.wrapper.trajectory_regions["right"]

    @right_region.setter
    def right_region(self, value: dict) -> None:
        self.set_region("right", value)

    @property
    def back_region(self) -> dict:
        return self.wrapper.trajectory_regions["back"]

    @back_region.setter
    def back_region(self, value: dict) -> None:
        """Override behind-bin goal disk (switch-collect train / test)."""
        self.set_region("back", value)

    @property
    def x_bound_max(self) -> float:
        return float(self.wrapper.x_bound_max)

    @x_bound_max.setter
    def x_bound_max(self, value: float) -> None:
        self.wrapper.x_bound_max = float(value)

    @property
    def use_arena_bounds(self) -> bool:
        return bool(self.wrapper.use_arena_bounds)

    @use_arena_bounds.setter
    def use_arena_bounds(self, value: bool) -> None:
        self.wrapper.use_arena_bounds = bool(value)

    @property
    def waypoint_layout(self) -> str:
        return str(self.wrapper.waypoint_layout)

    @waypoint_layout.setter
    def waypoint_layout(self, value: str) -> None:
        self.wrapper.waypoint_layout = str(value)

    @property
    def lidar_distance_threshold(self) -> float:
        return float(self.wrapper.lidar_distance_threshold)

    @lidar_distance_threshold.setter
    def lidar_distance_threshold(self, value: float) -> None:
        self.wrapper.lidar_distance_threshold = float(value)

    @property
    def lidar_h_half_fov_deg(self) -> float:
        return float(getattr(self.wrapper, "lidar_h_half_fov_deg", 90.0))

    @lidar_h_half_fov_deg.setter
    def lidar_h_half_fov_deg(self, value: float) -> None:
        self.wrapper.lidar_h_half_fov_deg = float(value)

    @property
    def include_contact_in_hs(self) -> bool:
        return bool(getattr(self.wrapper, "include_contact_in_hs", False))

    @include_contact_in_hs.setter
    def include_contact_in_hs(self, value: bool) -> None:
        self.wrapper.include_contact_in_hs = bool(value)

    @property
    def contact_hs(self) -> float:
        return float(getattr(self.wrapper, "contact_hs", -1.5))

    @contact_hs.setter
    def contact_hs(self, value: float) -> None:
        self.wrapper.contact_hs = float(value)

    @property
    def perp_offset(self) -> float:
        return float(self.wrapper.perp_offset)

    @perp_offset.setter
    def perp_offset(self, value: float) -> None:
        self.wrapper.perp_offset = float(value)

    @property
    def min_start_goal_dist(self) -> float:
        return float(self.wrapper.min_start_goal_dist)

    @min_start_goal_dist.setter
    def min_start_goal_dist(self, value: float) -> None:
        self.wrapper.min_start_goal_dist = float(value)

    @property
    def path_obstacle_layout(self) -> bool:
        return bool(getattr(self.wrapper, "path_obstacle_layout", False))

    @path_obstacle_layout.setter
    def path_obstacle_layout(self, value: bool) -> None:
        self.wrapper.path_obstacle_layout = bool(value)
        if value:
            self.wrapper.filter_lidar_to_active_obstacles = True

    @property
    def two_obstacle_prob(self) -> float:
        return float(getattr(self.wrapper, "two_obstacle_prob", 0.5))

    @two_obstacle_prob.setter
    def two_obstacle_prob(self, value: float) -> None:
        self.wrapper.two_obstacle_prob = float(value)

    @property
    def collect_controller(self) -> str:
        return str(getattr(self.wrapper, "collect_controller", "sf"))

    @collect_controller.setter
    def collect_controller(self, value: str) -> None:
        self.wrapper.collect_controller = str(value)

    @property
    def alternate_collect_controllers(self) -> bool:
        return bool(getattr(self.wrapper, "alternate_collect_controllers", False))

    @alternate_collect_controllers.setter
    def alternate_collect_controllers(self, value: bool) -> None:
        self.wrapper.alternate_collect_controllers = bool(value)

    @property
    def _collect_ep_toggle(self) -> int:
        return int(getattr(self.wrapper, "_collect_ep_toggle", 0))

    @_collect_ep_toggle.setter
    def _collect_ep_toggle(self, value: int) -> None:
        self.wrapper._collect_ep_toggle = int(value)

    @property
    def filter_lidar_to_active_obstacles(self) -> bool:
        return bool(getattr(self.wrapper, "filter_lidar_to_active_obstacles", False))

    @filter_lidar_to_active_obstacles.setter
    def filter_lidar_to_active_obstacles(self, value: bool) -> None:
        self.wrapper.filter_lidar_to_active_obstacles = bool(value)

    @property
    def arena_x_min(self) -> float:
        return float(self.wrapper.arena_x_min)

    @arena_x_min.setter
    def arena_x_min(self, value: float) -> None:
        self.wrapper.arena_x_min = float(value)

    @property
    def arena_x_max(self) -> float:
        return float(self.wrapper.arena_x_max)

    @arena_x_max.setter
    def arena_x_max(self, value: float) -> None:
        self.wrapper.arena_x_max = float(value)

    @property
    def arena_y_min(self) -> float:
        return float(self.wrapper.arena_y_min)

    @arena_y_min.setter
    def arena_y_min(self, value: float) -> None:
        self.wrapper.arena_y_min = float(value)

    @property
    def arena_y_max(self) -> float:
        return float(self.wrapper.arena_y_max)

    @arena_y_max.setter
    def arena_y_max(self, value: float) -> None:
        self.wrapper.arena_y_max = float(value)

    def set_arena_bounds(
        self,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> None:
        self.wrapper.arena_x_min = float(x_min)
        self.wrapper.arena_x_max = float(x_max)
        self.wrapper.arena_y_min = float(y_min)
        self.wrapper.arena_y_max = float(y_max)
        self.wrapper.use_arena_bounds = True

    def _arena_str(self) -> str:
        w = self.wrapper
        return (
            f"x[{w.arena_x_min:g},{w.arena_x_max:g}] "
            f"y[{w.arena_y_min:g},{w.arena_y_max:g}]"
        )

    def _apply_optional_pipeline_args(self, args) -> None:
        """New path-pipeline knobs. Missing attrs keep the original G1 defaults."""
        self.skip_arena_oob_from_buffer = bool(
            getattr(args, "skip_arena_oob_from_buffer", False)
        )
        thr = getattr(args, "lidar_distance_threshold", None)
        if thr is not None:
            self.wrapper.lidar_distance_threshold = float(thr)
        if getattr(args, "lidar_h_half_fov_deg", None) is not None:
            self.wrapper.lidar_h_half_fov_deg = float(args.lidar_h_half_fov_deg)
        if bool(getattr(args, "no_include_contact_in_hs", False)):
            self.wrapper.include_contact_in_hs = False
        elif bool(getattr(args, "include_contact_in_hs", False)):
            self.wrapper.include_contact_in_hs = True
        if getattr(args, "contact_hs", None) is not None:
            self.wrapper.contact_hs = float(args.contact_hs)
        yaw_limit = getattr(args, "yaw_limit", None)
        if yaw_limit is not None:
            yaw_limit = float(yaw_limit)
            self.wrapper.action_low = np.array(
                [0.0, -0.5, -yaw_limit], dtype=np.float32
            )
            self.wrapper.action_high = np.array(
                [0.8, 0.5, yaw_limit], dtype=np.float32
            )
        layout = getattr(args, "waypoint_layout", None)
        if layout:
            self.wrapper.waypoint_layout = str(layout)
        if bool(getattr(args, "use_arena_bounds", False)):
            self.wrapper.use_arena_bounds = True
            if getattr(args, "arena_x_min", None) is not None:
                self.wrapper.arena_x_min = float(args.arena_x_min)
            if getattr(args, "arena_x_max", None) is not None:
                self.wrapper.arena_x_max = float(args.arena_x_max)
            if getattr(args, "arena_y_min", None) is not None:
                self.wrapper.arena_y_min = float(args.arena_y_min)
            if getattr(args, "arena_y_max", None) is not None:
                self.wrapper.arena_y_max = float(args.arena_y_max)
        if getattr(args, "perp_offset", None) is not None:
            self.wrapper.perp_offset = float(args.perp_offset)
        if getattr(args, "min_start_goal_dist", None) is not None:
            self.wrapper.min_start_goal_dist = float(args.min_start_goal_dist)
        if bool(getattr(args, "path_obstacle_layout", False)):
            self.wrapper.path_obstacle_layout = True
            self.wrapper.filter_lidar_to_active_obstacles = True
        if getattr(args, "two_obstacle_prob", None) is not None:
            self.wrapper.two_obstacle_prob = float(args.two_obstacle_prob)
        if getattr(args, "obstacle_absent_prob", None) is not None:
            self.wrapper.obstacle_absent_prob = float(args.obstacle_absent_prob)
        layout_seed = getattr(args, "layout_seed", None)
        if layout_seed is not None:
            self.wrapper._rng = np.random.default_rng(int(layout_seed))
            print(f"[LatentHumanoidEnv] layout RNG seeded with --layout_seed={int(layout_seed)}")
        alt = bool(getattr(args, "alternate_collect", False))
        if bool(getattr(args, "no_alternate_collect", False)):
            alt = False
        self.wrapper.alternate_collect_controllers = alt
        if alt:
            self.wrapper.collect_controller = "waypoint"

    @property
    def obstacle_absent_prob(self) -> float:
        return float(self.wrapper.obstacle_absent_prob)

    @obstacle_absent_prob.setter
    def obstacle_absent_prob(self, value: float) -> None:
        """P(hide bin on each reset). Applies to buffer and formal train."""
        self.wrapper.obstacle_absent_prob = float(value)

    def _reset_timers(self) -> None:
        self._episode_sim_step = 0
        self._episode_visual_step = 0
        # Next encode boundary after t=0 (reset already encoded the t=0 frame).
        self._next_visual_time_s = self.visual_period_s

    def compute_waypoint_nav_action(self) -> np.ndarray:
        """Env-space (vx, vy, yaw_rate) from WaypointNavController toward current waypoint."""
        base_pos, base_quat = self.wrapper.get_robot_base_pose()
        cmd = self.waypoint_nav.compute_command(
            base_pos, base_quat, self.wrapper.waypoint
        )
        return self._clip_nav_cmd(cmd)

    def compute_safe_side_nav_action(self) -> np.ndarray:
        """Env-space action toward the closer fully-safe side waypoint (a_good).

        Picks (3.5, -0.5) or (3.5, -3.5) by XY distance. Does not touch the
        episode waypoint or the nominal controller state.
        """
        base_pos, base_quat = self.wrapper.get_robot_base_pose()
        xy = np.asarray(base_pos[:2], dtype=np.float64)
        pts = [
            np.asarray(p, dtype=np.float64).reshape(2)
            for p in self.safe_side_waypoints
        ]
        dists = [float(np.linalg.norm(xy - p)) for p in pts]
        target = pts[int(np.argmin(np.asarray(dists)))]
        cmd = self._safe_side_nav.compute_command(base_pos, base_quat, target)
        return self._clip_nav_cmd(cmd)

    def _clip_nav_cmd(self, cmd) -> np.ndarray:
        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        return np.clip(np.asarray(cmd, dtype=np.float32).reshape(3), low, high)

    def _pyhj_info(self, end_reason: str | None = None, stuck: bool = False) -> dict:
        """Scalar-only info with a fixed key set (required by PyHJ Batch assignment)."""
        skip_update = (
            end_reason == "out_of_bounds"
            and bool(getattr(self, "skip_arena_oob_from_buffer", False))
        )
        return {
            "episode_step": np.int32(self._episode_visual_step),
            "end_reason": np.int32(self._END_REASON_CODE.get(end_reason, 0)),
            "stuck": np.float32(1.0 if stuck else 0.0),
            "skip_update": np.float32(1.0 if skip_update else 0.0),
        }

    def _visual_to_uint8_hwc(self, visual: Any) -> np.ndarray:
        """Normalize wrapper visual to uint8 (H, W, 3)."""
        if isinstance(visual, torch.Tensor):
            arr = visual.detach().cpu().numpy()
        else:
            arr = np.asarray(visual)
        if arr.ndim != 3:
            raise ValueError(f"Expected HxWxC visual, got shape {arr.shape}")
        if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if np.issubdtype(arr.dtype, np.floating):
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        return arr[..., :3]

    def _hj_value(
        self,
        z: np.ndarray,
        action_env: np.ndarray | None = None,
    ) -> float | None:
        """Overlay HJ = Q(z, a) with a single critic forward (cheap).

        Prefer the action actually sent to the env (env-space → policy-space).
        Do **not** run critic action-sampling here — that path is for control /
        Bellman only, and calling it every video frame OOMs Isaac RTX on 24GB.
        """
        policy = self.policy_for_log
        if policy is None:
            return None
        try:
            with torch.no_grad():
                z_t = torch.as_tensor(z, dtype=torch.float32, device=self.device)
                if z_t.ndim == 1:
                    z_t = z_t.unsqueeze(0)
                if action_env is not None:
                    act_env = np.asarray(action_env, dtype=np.float64).reshape(1, -1)
                    act_pol = policy.map_action_inverse(act_env)
                    act_t = torch.as_tensor(
                        act_pol, dtype=torch.float32, device=self.device
                    )
                else:
                    # Reset frame: no action yet — score the zero policy action.
                    act_t = torch.zeros(
                        z_t.shape[0], 3, device=self.device, dtype=torch.float32
                    )
                q = policy.critic(z_t, act_t)
                return float(q.reshape(-1)[0].item())
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] HJ value compute failed: {exc}")
            return None

    def _log_rollout_metrics(
        self,
        l_val: float,
        hj_val: float | None,
        *,
        bump_env_step: bool = True,
        contact: float | None = None,
    ) -> None:
        """One wandb.log per env transition; x-axis is trainer/env_step."""
        # Always bump the shared step counter (trainer x-axis), even if wandb is off.
        if self.log_state is None:
            self.log_state = {"env_step": 0, "update": 0}
        if bump_env_step:
            self.log_state["env_step"] = int(self.log_state.get("env_step", 0)) + 1
        if not self.log_rollout_to_wandb:
            return
        try:
            import wandb

            if wandb.run is None:
                return
            step = int(self.log_state["env_step"])
            payload: dict[str, float] = {
                "trainer/env_step": float(step),
                "safety/l": float(l_val),
            }
            contact_v = contact
            if contact_v is None:
                contact_v = float(
                    getattr(self.wrapper, "_last_lidar_stats", {}).get(
                        "contact_collision", 0.0
                    )
                )
            payload["safety/contact"] = float(contact_v)
            if hj_val is not None and np.isfinite(hj_val):
                payload["safety/hj"] = float(hj_val)
            # Deterministic actor cmd stashed by exploration_noise wrapper (pre-noise).
            policy = self.policy_for_log
            act = getattr(policy, "last_clean_act_env", None) if policy is not None else None
            if act is not None:
                act = np.asarray(act, dtype=np.float64).reshape(-1)
                if act.size >= 3:
                    payload["actor_action/vx"] = float(act[0])
                    payload["actor_action/vy"] = float(act[1])
                    payload["actor_action/theta"] = float(act[2])
            q_nom = (
                getattr(policy, "last_q_nom", None) if policy is not None else None
            )
            if q_nom is not None and np.isfinite(q_nom):
                payload["switch/q_nom"] = float(q_nom)
            use_sf = (
                getattr(policy, "last_switch_use_sf", None)
                if policy is not None
                else None
            )
            if use_sf is not None:
                payload["switch/use_sf"] = float(use_sf)
            wandb.log(payload)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] rollout metric wandb log failed: {exc}")

    def _overlay_metrics_hwc(
        self, hwc: np.ndarray, hj_val: float | None, l_val: float | None
    ) -> np.ndarray:
        """Burn HJ / l text into top-left of an HxWxC uint8 frame."""
        from PIL import ImageDraw, ImageFont

        img = Image.fromarray(hwc)
        draw = ImageDraw.Draw(img)
        lines = []
        if hj_val is not None and np.isfinite(hj_val):
            lines.append(f"HJ {hj_val:.2f}")
        if l_val is not None and np.isfinite(l_val):
            lines.append(f"l {l_val:.2f}")
        if not lines:
            return hwc
        # Scale font with frame height (small wandb thumbs vs full RTX test videos).
        font_size = max(14, int(round(hwc.shape[0] * 0.045)))
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
            )
        except OSError:
            try:
                font = ImageFont.load_default(size=font_size)
            except TypeError:
                font = ImageFont.load_default()
        line_h = font_size + max(2, font_size // 6)
        y = 4
        x = 4
        for line in lines:
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1)):
                draw.text((x + dx, y + dy), line, fill=(0, 0, 0), font=font)
            draw.text((x, y), line, fill=(255, 60, 60), font=font)
            y += line_h
        return np.asarray(img, dtype=np.uint8)

    def _resize_frame_chw(
        self,
        visual: Any,
        hj_val: float | None = None,
        l_val: float | None = None,
    ) -> np.ndarray:
        """Return downscaled uint8 frame as (C, H, W) for wandb.Video."""
        hwc = self._visual_to_uint8_hwc(visual)
        th, tw = self.wandb_video_size
        img = Image.fromarray(hwc)
        img = img.resize((tw, th), Image.BILINEAR)
        hwc_small = np.asarray(img, dtype=np.uint8)
        hwc_small = self._overlay_metrics_hwc(hwc_small, hj_val, l_val)
        return np.transpose(hwc_small, (2, 0, 1))

    def _start_episode_recording(self) -> None:
        self._episode_frames = []
        # Record every N finished-episode boundaries (N=self.wandb_video_every).
        # e.g. N=5 → episodes starting at finished=0,5,10,...
        every = int(self.wandb_video_every)
        self._record_this_episode = every > 0 and (
            int(self._finished_episodes) % every == 0
        )

    def _append_frame(
        self,
        obs: dict[str, Any],
        hj_val: float | None = None,
        l_val: float | None = None,
    ) -> None:
        if not self._record_this_episode:
            return
        try:
            self._episode_frames.append(
                self._resize_frame_chw(obs["visual"], hj_val=hj_val, l_val=l_val)
            )
        except Exception as exc:  # noqa: BLE001 — never break training for logging
            print(f"[WARN] wandb frame append failed: {exc}")

    def _log_episode_video(self, end_reason: str | None) -> None:
        if not self._record_this_episode or not self._episode_frames:
            return
        try:
            import wandb

            if wandb.run is None:
                return
            frames = np.stack(self._episode_frames, axis=0)  # (T, C, H, W)
            ep_id = self._finished_episodes
            payload = {
                "rollout/video": wandb.Video(
                    frames,
                    fps=max(1, int(round(self.visual_fps))),
                    format="gif",
                ),
                "rollout/episode": ep_id,
                "rollout/end_reason": self._END_REASON_CODE.get(end_reason, 0),
                "rollout/visual_steps": self._episode_visual_step,
            }
            if self.log_state is not None:
                payload["trainer/env_step"] = float(self.log_state.get("env_step", 0))
            wandb.log(payload)
            print(
                f"[LatentHumanoidEnv] uploaded wandb video for episode {ep_id} "
                f"({len(self._episode_frames)} frames, reason={end_reason})"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] wandb video upload failed: {exc}")
        finally:
            self._episode_frames = []
            self._record_this_episode = False

    def _current_switch_use_sf(self) -> bool:
        """True iff this step executed SF (Q(a_nom)/HJ < 0)."""
        policy = self.policy_for_log
        if policy is None:
            return False
        use_sf = getattr(policy, "last_switch_use_sf", None)
        if use_sf is not None:
            try:
                return bool(float(use_sf) > 0.5)
            except (TypeError, ValueError):
                return bool(use_sf)
        q_nom = getattr(policy, "last_q_nom", None)
        if q_nom is not None and np.isfinite(q_nom):
            return bool(float(q_nom) < 0.0)
        return False

    def _start_switch_traj_recording(self, reset_info: dict) -> None:
        self._switch_traj_xy = []
        self._switch_traj_use_sf = []
        mode = str(
            (reset_info or {}).get(
                "collect_controller", getattr(self.wrapper, "collect_controller", "sf")
            )
        )
        self._record_switch_traj_ep = bool(self.record_switch_traj) and mode == "switch"
        if not self._record_switch_traj_ep:
            return
        try:
            xy = self.wrapper.get_robot_xy_local()
            self._switch_traj_xy.append(np.asarray(xy, dtype=np.float64).reshape(2).copy())
            self._switch_traj_use_sf.append(False)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] switch traj start failed: {exc}")
            self._record_switch_traj_ep = False

    def _append_switch_traj_xy(self) -> None:
        if not self._record_switch_traj_ep:
            return
        try:
            xy = self.wrapper.get_robot_xy_local()
            self._switch_traj_xy.append(np.asarray(xy, dtype=np.float64).reshape(2).copy())
            self._switch_traj_use_sf.append(self._current_switch_use_sf())
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] switch traj append failed: {exc}")

    def _finish_switch_traj(self, end_reason: str | None) -> None:
        if not self._record_switch_traj_ep:
            self._switch_traj_xy = []
            self._switch_traj_use_sf = []
            return
        self._record_switch_traj_ep = False
        layout = getattr(self, "_last_reset_layout", None) or {}
        try:
            from env.isaac.switch_traj_plot import save_switch_traj_png

            out_dir = Path(self.switch_traj_dir) if self.switch_traj_dir else Path(
                "runs/sac_hj_humanoid_path/switch_traj"
            )
            ep_id = int(self._finished_episodes)
            reason = end_reason or "ongoing"
            out_path = out_dir / f"switch_ep{ep_id:04d}_{reason}.png"
            arena = None
            if bool(getattr(self.wrapper, "use_arena_bounds", False)):
                arena = (
                    float(self.wrapper.arena_x_min),
                    float(self.wrapper.arena_x_max),
                    float(self.wrapper.arena_y_min),
                    float(self.wrapper.arena_y_max),
                )
            r = float(self.failure_set_radius)
            if r <= 0:
                r = float(self.wrapper.lidar_distance_threshold)
            png = save_switch_traj_png(
                out_path,
                np.asarray(self._switch_traj_xy, dtype=np.float64),
                layout.get("waypoints"),
                layout.get("obstacle_xys") or [],
                failure_radius=r,
                arena=arena,
                end_reason=reason,
                episode_id=ep_id,
                use_sf=np.asarray(self._switch_traj_use_sf, dtype=bool),
            )
            print(f"[LatentHumanoidEnv] saved switch traj {png}")
            try:
                import wandb

                if wandb.run is not None:
                    payload = {
                        "rollout/switch_traj": wandb.Image(str(png)),
                        "rollout/episode": ep_id,
                    }
                    if self.log_state is not None:
                        payload["trainer/env_step"] = float(
                            self.log_state.get("env_step", 0)
                        )
                    wandb.log(payload)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] switch traj wandb log failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] switch traj plot failed: {exc}")
        finally:
            self._switch_traj_xy = []
            self._switch_traj_use_sf = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        fixed_layout = None if options is None else options.get("fixed_layout")
        reset_info = self.wrapper.reset_scene(seed=seed, fixed_layout=fixed_layout)
        self._reset_timers()
        self.waypoint_nav.reset()
        self._safe_side_nav.reset()
        self._start_episode_recording()
        obs = self.wrapper.get_raw_obs()
        z = self.encode(obs)
        self._reset_wm_history(obs)
        l_val = float(self.wrapper.calculate_cost())
        hj_val = self._hj_value(z)
        # Reset frame: same env_step index, do not bump (no transition yet).
        self._log_rollout_metrics(l_val, hj_val, bump_env_step=False)
        self._append_frame(obs, hj_val=hj_val, l_val=l_val)
        if reset_info is not None:
            print(
                f"[LatentHumanoidEnv] reset robot_xy={reset_info.get('robot_xy')} "
                f"n_obstacles={reset_info.get('n_obstacles')} "
                f"obstacle_xys={reset_info.get('obstacle_xys')} "
                f"collect={reset_info.get('collect_controller')} "
                f"wp_names={reset_info.get('waypoint_region_names')} "
                f"waypoints={reset_info.get('waypoints')}"
            )
        info = self._pyhj_info(end_reason=None, stuck=False)
        # Keep layout on the side for multi-mode eval (not in PyHJ Batch schema).
        self._last_reset_layout = {
            "blue_bin_xy": reset_info.get("blue_bin_xy"),
            "obstacle_present": bool(reset_info.get("obstacle_present", True)),
            "n_obstacles": int(reset_info.get("n_obstacles", 0)),
            "obstacle_xys": reset_info.get("obstacle_xys"),
            "collect_controller": reset_info.get("collect_controller"),
            "waypoints": np.asarray(reset_info["waypoints"], dtype=np.float64).copy(),
            "waypoint_region_names": list(reset_info.get("waypoint_region_names", [])),
            "robot_xy": np.asarray(reset_info["robot_xy"], dtype=np.float64).copy(),
        }
        self._start_switch_traj_recording(reset_info or {})
        return z, info

    def step(self, action):
        """Hold ``action`` across sim substeps until the next 15 fps visual sample."""
        import time as _time

        # Continuous margin: smaller = more dangerous → aggregate with min.
        h_s = float("inf")
        stuck = False
        end_reason = None
        step_info: dict[str, Any] = {}
        contact_any = 0.0
        dbg = bool(self.debug_step_timing) and int(self._debug_steps_left) > 0
        t_enter = _time.time() if dbg else 0.0
        if dbg:
            print(
                f"[LatentHumanoidEnv] step enter visual={self._episode_visual_step} "
                f"sim={self._episode_sim_step}",
                flush=True,
            )

        # Run control-rate physics until we hit the next visual sample time
        # (same gating idea as collect_humanoid_dataset.subsample_frame_indices).
        while True:
            if dbg and self._episode_sim_step == 0:
                print("[LatentHumanoidEnv] first apply_velocity_command...", flush=True)
            _, _isaac_terminated, _, step_info = self.wrapper.apply_velocity_command(action)
            del _isaac_terminated
            self._episode_sim_step += 1
            sim_time_s = self._episode_sim_step * self.sim_dt
            if dbg and self._episode_sim_step == 1:
                print(
                    f"[LatentHumanoidEnv] first sim step done in {_time.time() - t_enter:.1f}s",
                    flush=True,
                )

            h_s = min(h_s, float(self.wrapper.calculate_cost()))
            if float(self.wrapper._last_lidar_stats.get("contact_collision", 0.0)) > 0.5:
                contact_any = 1.0
            stuck = stuck or bool(step_info.get("stuck", False))

            if self.wrapper.advance_waypoint_if_reached():
                end_reason = "all_waypoints"
                break
            if stuck:
                end_reason = "stuck"
                break
            # Soft walls: cut flee trajectories. Not a safety failure.
            if self.wrapper.is_out_of_bounds():
                end_reason = "out_of_bounds"
                break
            if self._episode_sim_step >= self.max_episode_sim_steps:
                end_reason = "max_steps"
                break

            if sim_time_s + 1e-12 >= self._next_visual_time_s:
                self._next_visual_time_s += self.visual_period_s
                break

        self._episode_visual_step += 1
        if dbg:
            print(
                f"[LatentHumanoidEnv] physics done sim={self._episode_sim_step} "
                f"({_time.time() - t_enter:.1f}s); reading RTX+encode...",
                flush=True,
            )
        obs = self.wrapper.get_raw_obs()
        self._push_wm_history(obs, action)
        z_next = self.encode(obs)
        # Q(z, a_executed) — same cost as the old actor-based overlay.
        hj_val = self._hj_value(z_next, action_env=np.asarray(action, dtype=np.float64))
        # out_of_bounds: truncate only (stop collecting sparse flee data). Do NOT
        # rewrite h_s — OOB is not a collision / not a safety failure label.
        self._log_rollout_metrics(h_s, hj_val, bump_env_step=True, contact=contact_any)
        self._append_frame(obs, hj_val=hj_val, l_val=h_s)
        self._append_switch_traj_xy()
        if dbg:
            print(
                f"[LatentHumanoidEnv] step exit in {_time.time() - t_enter:.1f}s "
                f"h_s={h_s:.3f}",
                flush=True,
            )
            self._debug_steps_left = max(0, int(self._debug_steps_left) - 1)

        terminated = False
        truncated = end_reason is not None
        if truncated:
            xy = self.wrapper.get_robot_xy_local()
            if end_reason == "out_of_bounds":
                if bool(getattr(self.wrapper, "use_arena_bounds", False)):
                    extra = (
                        f" (arena xy=[{xy[0]:.3f},{xy[1]:.3f}] "
                        f"{self._arena_str()}, truncate-only"
                        f"{', skip_update' if self.skip_arena_oob_from_buffer else ''})"
                    )
                elif self.wrapper.is_out_of_x_bounds():
                    extra = (
                        f" (x={xy[0]:.3f}>={self.wrapper.x_bound_max}, truncate-only)"
                    )
                else:
                    dy = abs(float(xy[1]) - float(self.wrapper.y_center))
                    extra = (
                        f" (|y-({self.wrapper.y_center})|={dy:.3f}"
                        f">{self.wrapper.y_bound}, truncate-only)"
                    )
            else:
                extra = f" robot_xy=[{xy[0]:.3f},{xy[1]:.3f}]"
            print(
                f"[LatentHumanoidEnv] episode end reason={end_reason}{extra} "
                f"visual_steps={self._episode_visual_step} "
                f"sim_steps={self._episode_sim_step}"
            )
            self._finished_episodes += 1
            self._log_episode_video(end_reason)
            self._finish_switch_traj(end_reason)
        return z_next, h_s, terminated, truncated, self._pyhj_info(end_reason, stuck)

    def _wm_num_hist(self) -> int:
        return max(1, int(getattr(self.wm, "num_hist", 1)))

    def _split_raw_obs(self, obs: dict[str, Any] | tuple | list):
        if isinstance(obs, dict):
            return obs["visual"], obs["proprio"]
        if isinstance(obs, (tuple, list)) and len(obs) == 2:
            return obs[0], obs[1]
        raise ValueError(f"Unexpected obs type: {type(obs)}")

    def _copy_raw_obs(self, obs: dict[str, Any] | tuple | list) -> dict[str, np.ndarray]:
        visual, proprio = self._split_raw_obs(obs)
        if isinstance(visual, torch.Tensor):
            visual_np = visual.detach().cpu().numpy()
        else:
            visual_np = np.array(visual, copy=True)
        if isinstance(proprio, torch.Tensor):
            proprio_np = proprio.detach().cpu().numpy().astype(np.float32, copy=True)
        else:
            proprio_np = np.array(proprio, dtype=np.float32, copy=True)
        return {"visual": visual_np, "proprio": proprio_np}

    def _visual_to_normalized_chw(self, visual) -> np.ndarray:
        """Match ``encode``: HWC → CHW float in [-1, 1]."""
        if isinstance(visual, torch.Tensor):
            visual_np = visual.permute(2, 0, 1).float().cpu().numpy()
            if visual_np.max() > 1.0:
                visual_np = visual_np / 255.0
        else:
            visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32)
            visual_np = visual_np / 255.0
        return np.ascontiguousarray((visual_np - 0.5) / 0.5, dtype=np.float32)

    def _flatten_latent_obs(self, lat: dict[str, torch.Tensor]) -> np.ndarray:
        z_vis = lat["visual"].reshape(1, -1)
        if self.with_proprio:
            z_prop = lat["proprio"].reshape(z_vis.shape[0], -1)
            z = torch.cat([z_vis, z_prop], dim=-1)
        else:
            z = z_vis
        return z.squeeze(0).cpu().numpy()

    def _reset_wm_history(self, obs: dict[str, Any] | tuple | list) -> None:
        """Repeat the first frame so the predictor has ``num_hist`` context."""
        n = self._wm_num_hist()
        self._obs_hist = [self._copy_raw_obs(obs) for _ in range(n)]
        act_dim = int(np.prod(self.action_space.shape))
        zero = np.zeros((act_dim,), dtype=np.float32)
        self._act_hist = [zero.copy() for _ in range(max(0, n - 1))]

    def _push_wm_history(self, obs: dict[str, Any] | tuple | list, action) -> None:
        """After a step: new obs is current; executed action belongs to the previous frame."""
        n = self._wm_num_hist()
        self._obs_hist.append(self._copy_raw_obs(obs))
        self._obs_hist = self._obs_hist[-n:]
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        self._act_hist.append(a)
        keep = max(0, n - 1)
        self._act_hist = self._act_hist[-keep:] if keep > 0 else []

    def predict_next_latent(self, action_env) -> np.ndarray:
        """WM one-step look-ahead: ``z_{t+1}`` from history + candidate env action.

        Same flattening as ``encode`` (critic space). Uses ``encode`` + ``predict``
        on a ``num_hist`` window, with ``action_env`` as the action at the last
        (current) frame — matching ``scripts/pred_recon_wm_episode.py``.
        """
        if self.wm.predictor is None:
            raise RuntimeError("World model has no predictor; cannot look ahead.")
        n = self._wm_num_hist()
        if len(self._obs_hist) < n:
            raise RuntimeError("WM history is empty; call reset() first.")

        action = np.asarray(action_env, dtype=np.float32).reshape(-1)
        acts = list(self._act_hist[-max(0, n - 1) :])
        while len(acts) < n - 1:
            acts.insert(0, np.zeros_like(action))
        acts.append(action)

        vis_list = []
        prop_list = []
        for frame in self._obs_hist[-n:]:
            vis_list.append(self._visual_to_normalized_chw(frame["visual"]))
            prop_list.append(np.asarray(frame["proprio"], dtype=np.float32).reshape(-1))

        vis_t = torch.from_numpy(np.stack(vis_list, axis=0)).unsqueeze(0).to(self.device)
        prop_t = torch.from_numpy(np.stack(prop_list, axis=0)).unsqueeze(0).to(self.device)
        act_t = torch.from_numpy(np.stack(acts, axis=0)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            z = self.wm.encode({"visual": vis_t, "proprio": prop_t}, act_t)
            z_pred = self.wm.predict(z)
            z_next = z_pred[:, -1:, ...]
            z_obs, _ = self.wm.separate_emb(z_next)
            return self._flatten_latent_obs(z_obs)

    def encode(self, obs: dict[str, Any] | tuple | list) -> np.ndarray:
        """Encode visual + proprio into a flat latent vector via the world model."""
        visual, proprio = self._split_raw_obs(obs)

        with torch.no_grad():
            vis_np = self._visual_to_normalized_chw(visual)
            vis_t = torch.from_numpy(vis_np).unsqueeze(0).unsqueeze(1).to(self.device)
            if isinstance(proprio, torch.Tensor):
                prop_t = proprio.unsqueeze(0).unsqueeze(1).float().to(self.device)
            else:
                prop_t = (
                    torch.from_numpy(np.asarray(proprio, dtype=np.float32))
                    .unsqueeze(0)
                    .unsqueeze(1)
                    .to(self.device)
                )
            lat = self.wm.encode_obs({"visual": vis_t, "proprio": prop_t})
            return self._flatten_latent_obs(lat)

    def calculate_cost(self) -> float:
        return self.wrapper.calculate_cost()

    def close(self):
        self.wrapper.close()
