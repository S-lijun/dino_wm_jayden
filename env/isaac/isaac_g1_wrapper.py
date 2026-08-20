"""Isaac Lab G1 locomotion wrapper for latent safety-filter training.

Must be imported only after ``AppLauncher`` has started the simulation app.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Sequence

import numpy as np
import torch

from env.isaac.waypoint_utils import (
    DEFAULT_ARENA_BOUNDS,
    DEFAULT_DANGER_RADIUS_M,
    DEFAULT_MIN_START_GOAL_DIST,
    DEFAULT_PERP_OFFSET_M,
    DEFAULT_TRAJECTORY_REGIONS,
    DEFAULT_TRAJECTORY_REGION_SEQUENCE,
    PASS_SIDE_CYCLE,
    PASS_SIDE_TRAIN,
    compute_perp_sidestep_xy,
    generate_random_waypoint_sequence,
    sample_point_in_half_disk,
    sample_point_in_region,
    sample_perp_points_on_segment,
    sample_start_goal_perp_path,
    waypoints_to_list,
    quat_to_yaw,
)

# Obstacle scene keys registered on env_cfg.scene (see data_collection_obstacles.py).
# LiDAR: one RayCaster per mesh, then merge_ray_hits_multi → lidar_min_distance.
OBSTACLE_SPECS: dict[str, dict[str, Any]] = {
    "blue_bin_0": {
        "default_z": 0.5,
        "rot": (0.5, 0.5, 0.5, 0.5),
        "spawn_xy": (3.5, -2.0),
    },
}

_BIN_DEFAULT_Z = 0.5
_BIN_DEFAULT_ROT = (0.5, 0.5, 0.5, 0.5)


def _bin_spec(index: int) -> dict[str, Any]:
    name = f"blue_bin_{index}"
    if name in OBSTACLE_SPECS:
        return dict(OBSTACLE_SPECS[name])
    return {
        "default_z": _BIN_DEFAULT_Z,
        "rot": _BIN_DEFAULT_ROT,
        "spawn_xy": (3.5 + 0.05 * index, -2.0),
    }

HIDDEN_OBSTACLE_POS = (100.0, 100.0, -10.0)
DEFAULT_SENSOR_IMG_RES = (640, 480)  # portrait (height, width) before CCW rotation


def landscape_output_size(img_res: tuple[int, int]) -> tuple[int, int]:
    """Portrait sensor resolution → landscape (H, W) after CCW 90° rotation."""
    return (img_res[1], img_res[0])


# Backward-compatible alias for default landscape output (480×640).
VISUAL_SIZE = landscape_output_size(DEFAULT_SENSOR_IMG_RES)


def merge_ray_hits_multi(
    origin_np: np.ndarray,
    hits_list: list[np.ndarray],
    max_d: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge k (N,3) world hit arrays from sensors with the same origin."""
    if len(hits_list) == 0:
        raise ValueError("hits_list must contain at least one lidar hit array.")

    diffs = [hits - origin_np for hits in hits_list]
    ranges_all = [np.linalg.norm(d, axis=1) for d in diffs]
    finite_all = [
        np.isfinite(hits).all(axis=1) & (r > 1e-4) & (r < max_d * 0.999)
        for hits, r in zip(hits_list, ranges_all)
    ]

    n = hits_list[0].shape[0]
    merged = np.full_like(diffs[0], np.inf)
    best_r = np.full(n, np.inf)

    for diff, r, valid in zip(diffs, ranges_all, finite_all):
        take = valid & (r < best_r)
        merged[take] = diff[take]
        best_r[take] = r[take]

    diff_w = merged
    ranges = np.linalg.norm(diff_w, axis=1)
    ranges_xy = np.linalg.norm(diff_w[:, :2], axis=1)
    return diff_w, ranges, ranges_xy


def compute_lidar_min_range_labels(
    origin_np: np.ndarray,
    hits_list: list[np.ndarray],
    max_d: float,
) -> tuple[float, float]:
    """Match DataCollection_test.py ``lidar_min_range_nonzero_m`` / ``_xy``."""
    diff_w, ranges, ranges_xy = merge_ray_hits_multi(origin_np, hits_list, max_d)
    valid_ray = np.isfinite(diff_w).all(axis=1) & (ranges > 1e-4) & (ranges < max_d * 0.999)
    positive_ranges = ranges[valid_ray]
    lidar_min_range_nonzero_m = (
        float(np.min(positive_ranges)) if positive_ranges.size > 0 else float("nan")
    )
    positive_ranges_xy = ranges_xy[valid_ray]
    lidar_min_range_xy_nonzero_m = (
        float(np.min(positive_ranges_xy)) if positive_ranges_xy.size > 0 else float("nan")
    )
    return lidar_min_range_nonzero_m, lidar_min_range_xy_nonzero_m


def _object_to_mesh_path(object_name: str, env_prim_root: str) -> str:
    obj_name = object_name.strip()
    if obj_name.startswith("/"):
        return obj_name
    return f"{env_prim_root.rstrip('/')}/{obj_name}"


def _resize_rgb(rgb_np: np.ndarray, size: tuple[int, int] = VISUAL_SIZE) -> np.ndarray:
    """Resize RGB array to (H, W, 3) uint8."""
    from PIL import Image

    if rgb_np.dtype != np.uint8:
        rgb_np = (np.clip(rgb_np, 0.0, 1.0) * 255).astype(np.uint8)
    img = Image.fromarray(rgb_np)
    img = img.resize((size[1], size[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


class IsaacG1Wrapper:
    """Low-level Isaac Lab G1 env: PPO locomotion + LiDAR + contact safety."""

    TASK = "Isaac-Velocity-Flat-G1-v0"
    RL_LIBRARY = "rsl_rl"

    def __init__(
        self,
        args_cli,
        *,
        visual_mode: str | None = None,
        img_res: tuple[int, int] = (640, 480),
        env_prim_root: str = "/World/envs/env_0",
        lidar_distance_threshold: float = 1.0,
        collision_force_threshold: float = 0.1,
        stuck_contact_steps: int = 50,
        waypoint_stop_thresh: float = 0.1,
        # Soft corridor about aisle centerline y_center (waypoints/bin at y=-2).
        y_bound: float = 0.0,
        y_center: float = -2.0,
        # Far end of the aisle: x >= this truncates (same as y OOB).
        # Bin at x=3.5 → default wall at 4.5 (~1m past bin).
        x_bound_max: float = 4.5,
        trajectory_regions: dict[str, dict[str, Any]] | None = None,
        trajectory_region_sequence: Sequence[str | tuple[str, ...]] | None = None,
        max_speed: float = 0.5,
        demos_dir: str | None = None,
        blue_bin_xy: tuple[float, float] | None = None,
    ):
        self.args_cli = args_cli
        if visual_mode is None:
            visual_mode = getattr(args_cli, "visual_mode", "depth_rgb")
        self.visual_mode = visual_mode
        self.collect_visual = self.visual_mode != "off"
        self.img_res = img_res
        self.visual_output_size = landscape_output_size(img_res)
        if blue_bin_xy is None:
            blue_bin_xy = (
                float(getattr(args_cli, "bin_x", OBSTACLE_SPECS["blue_bin_0"]["spawn_xy"][0])),
                float(getattr(args_cli, "bin_y", OBSTACLE_SPECS["blue_bin_0"]["spawn_xy"][1])),
            )
        self._blue_bin_xy = blue_bin_xy
        # Buffer: keep bin fixed. Training: resample y on bin x (3.5) within obstacle_y_range.
        self._blue_bin_xy_fixed = (float(blue_bin_xy[0]), float(blue_bin_xy[1]))
        self.randomize_obstacle = False
        self.obstacle_y_range = (-2.5, -1.5)
        # Independent of y-jitter: with this prob hide the bin entirely on reset.
        self.obstacle_absent_prob = 0.0
        self.two_obstacle_prob = 0.5
        self.obstacle_present = True
        self.path_obstacle_layout = False
        self.filter_lidar_to_active_obstacles = False
        self._max_n_obstacles = max(1, int(getattr(args_cli, "max_n_obstacles", 1)))
        self._obstacle_names = [f"blue_bin_{i}" for i in range(self._max_n_obstacles)]
        self._obstacle_specs = {
            name: _bin_spec(i) for i, name in enumerate(self._obstacle_names)
        }
        self._obstacle_xy_list: list[np.ndarray] = []
        self._n_obstacles = 0
        # Collect controller for the current episode: "sf" | "waypoint" | "switch".
        self.collect_controller = "sf"
        self.alternate_collect_controllers = False
        self._collect_ep_toggle = 0
        self._n_scene_resets = 0
        self.env_prim_root = env_prim_root
        self.lidar_distance_threshold = lidar_distance_threshold
        # Half-width of the forward cone used for l / h_s (deg). Inside the
        # cone, l is the true min range; outside, treat as no-hit (l=2).
        # Default ±90° keeps the original aisle pipeline. Path uses ±60°.
        self.lidar_h_half_fov_deg = float(
            getattr(args_cli, "lidar_h_half_fov_deg", 90.0)
        )
        self.collision_force_threshold = collision_force_threshold
        # Path pipeline: non-foot obstacle contact folds into h_s / l. Off for aisle.
        self.include_contact_in_hs = False
        self.contact_hs = -1.5
        # Hold a_good sidestep target this many HJ steps after first HJ<0.
        self.sidestep_hold_steps = 20
        self._sidestep_xy: np.ndarray | None = None
        self._sidestep_hold = 0
        self.stuck_contact_steps = int(stuck_contact_steps)
        self.waypoint_stop_thresh = float(waypoint_stop_thresh)
        # If True, geometrically passing the last waypoint also completes it
        # (needed for switch-collect train: pass-side left|right is the goal).
        # Test keeps False so the terminal back disk still requires proximity.
        self.advance_passed_terminal = False
        # y_bound <= 0 disables the soft |y - y_center| corridor (default: disabled).
        self.y_bound = float(getattr(args_cli, "y_bound", y_bound))
        self.y_center = float(getattr(args_cli, "y_center", y_center))
        self.x_bound_max = float(getattr(args_cli, "x_bound_max", x_bound_max))
        # Rectangular arena (new path pipeline). Off by default so the original
        # aisle walls (y_bound / x_bound_max) stay unchanged.
        self.use_arena_bounds = False
        self.arena_x_min = float(DEFAULT_ARENA_BOUNDS[0])
        self.arena_x_max = float(DEFAULT_ARENA_BOUNDS[1])
        self.arena_y_min = float(DEFAULT_ARENA_BOUNDS[2])
        self.arena_y_max = float(DEFAULT_ARENA_BOUNDS[3])
        # "regions" = original start/front/left|right. "start_goal_perp" = new path.
        self.waypoint_layout = "regions"
        self.perp_offset = float(DEFAULT_PERP_OFFSET_M)
        self.min_start_goal_dist = float(DEFAULT_MIN_START_GOAL_DIST)
        self.arena_sample_margin = 0.5
        self._spawn_yaw: float | None = None
        # Env-space velocity clip. Original pipeline: yaw in [-0.5, 0.5].
        self.action_low = np.array([0.0, -0.5, -0.5], dtype=np.float32)
        self.action_high = np.array([0.8, 0.5, 0.5], dtype=np.float32)
        # Copy defaults (start disk r=1, front disk r=0.5, left|right r=0.5, middle point).
        if trajectory_regions is not None:
            self.trajectory_regions = trajectory_regions
        else:
            self.trajectory_regions = {
                k: {
                    **v,
                    **(
                        {"center": np.asarray(v["center"], dtype=np.float64).copy()}
                        if "center" in v
                        else {}
                    ),
                }
                for k, v in DEFAULT_TRAJECTORY_REGIONS.items()
            }
        self.trajectory_region_sequence = (
            trajectory_region_sequence
            if trajectory_region_sequence is not None
            else DEFAULT_TRAJECTORY_REGION_SEQUENCE
        )
        # Buffer: cycle left → right → middle. Train/test: left/right goals only.
        self.alternate_left_right = True
        # True only during buffer warm-up collision demos; test should set False.
        self.include_middle_pass = True
        # If set to "left"/"right"/"middle", always use that pass side (skip cycle).
        self.force_pass_side: str | None = None
        # Formal-train option: alternate spawn half-disk and couple pass side.
        # left hemisphere (y>=center) → left region; right (y<=center) → right.
        self.spawn_hemisphere_pass: bool = False
        self._pass_side_toggle = 0  # indexes pass-side cycle
        self._hemisphere_toggle = 0  # indexes left/right hemisphere alternation
        self.max_speed = max_speed

        if demos_dir is None:
            demos_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../IsaacLab/scripts/demos",
            )
        self.demos_dir = os.path.abspath(demos_dir)
        if self.demos_dir not in sys.path:
            sys.path.insert(0, self.demos_dir)

        from data_collection_obstacles import add_blue_bin
        import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
        from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
        from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
            G1FlatEnvCfg_PLAY,
        )
        from isaaclab.sensors import ContactSensorCfg
        from isaaclab.sensors.camera import CameraCfg
        from isaaclab.sensors.ray_caster import RayCasterCfg, RayCasterCameraCfg, patterns
        from visual_obs_utils import (
            build_depth_camera_cfgs,
            build_rtx_camera_cfg,
            depth_to_rgb,
            lidar_ranges_to_rgb,
            merge_depth_maps_multi,
            resize_rgb,
        )
        from lab_scene_utils import (
            default_raycast_mesh_paths,
            load_lab_scene_usd,
            rotate_sensor_ccw_to_landscape,
        )
        self._rotate_sensor_ccw_to_landscape = rotate_sensor_ccw_to_landscape
        self._resize_rgb = resize_rgb
        self._merge_depth_maps_multi = merge_depth_maps_multi
        self._depth_to_rgb = depth_to_rgb
        self._lidar_ranges_to_rgb = lidar_ranges_to_rgb
        import isaaclab.sim as sim_utils
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom

        self._UsdGeom = UsdGeom
        self._Gf = Gf
        self._Sdf = Sdf
        self._omni_usd = omni.usd

        self._depth_cam_names: list[str] = []

        # depth_rgb: lab meshes needed for RayCasterCamera.
        # rtx_rgb: load lab BEFORE env (matches test_gs_rtx_smoke); loading after
        # often leaves CameraCfg annotator with invalid overscan on first RGB read.
        if self.visual_mode in ("depth_rgb", "rtx_rgb"):
            load_lab_scene_usd(demos_dir=self.demos_dir)
        if self.visual_mode == "depth_rgb":
            self._lidar_mesh_paths = default_raycast_mesh_paths(
                env_prim_root,
                obstacle_names=tuple(self._obstacle_names),
                include_lab_scene=True,
            )
        else:
            self._lidar_mesh_paths = default_raycast_mesh_paths(
                env_prim_root,
                obstacle_names=tuple(self._obstacle_names),
                include_lab_scene=False,
            )
        self._lidar_sensor_names = [f"lidar_{i}" for i in range(len(self._lidar_mesh_paths))]
        self._obstacle_lidar_names: dict[str, str] = {}
        for i, mesh_path in enumerate(self._lidar_mesh_paths):
            path_s = str(mesh_path).rstrip("/")
            for obs_name in self._obstacle_names:
                if path_s.endswith("/" + obs_name) or path_s.endswith(obs_name):
                    self._obstacle_lidar_names[obs_name] = self._lidar_sensor_names[i]

        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(self.TASK, args_cli)
        checkpoint = get_published_pretrained_checkpoint(self.RL_LIBRARY, self.TASK)

        env_cfg = G1FlatEnvCfg_PLAY()
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 100000
        env_cfg.curriculum = None
        env_cfg.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        env_cfg.decimation = 1
        env_cfg.sim.render_interval = 1
        env_cfg.terminations.base_contact = None

        use_path_spawn = bool(getattr(args_cli, "path_obstacle_layout", False)) or (
            str(getattr(args_cli, "waypoint_layout", "regions")) == "start_goal_perp"
        )
        for i, name in enumerate(self._obstacle_names):
            spec = self._obstacle_specs[name]
            # Path layout teleports bins every reset onto the start–goal
            # perpendicular. Do not spawn bin_0 at the old aisle (3.5,-2).
            if i == 0 and not use_path_spawn:
                xy = spec["spawn_xy"]
                z = spec["default_z"]
            else:
                xy = (HIDDEN_OBSTACLE_POS[0], HIDDEN_OBSTACLE_POS[1])
                z = HIDDEN_OBSTACLE_POS[2]
            add_blue_bin(env_cfg, pos=(float(xy[0]), float(xy[1]), float(z)), index=i)

        env_cfg.scene.robot_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*link.*",
            update_period=0.0,
            debug_vis=False,
            filter_prim_paths_expr=[],
        )

        self.lidar_fps = 7.0
        self.lidar_period_s = 1.0 / self.lidar_fps

        if self.visual_mode == "rtx_rgb":
            # update_period=0 → refresh every sim step (avoids empty first annotator frame).
            env_cfg.scene.camera = build_rtx_camera_cfg(
                img_res=img_res,
                update_period_s=0.0,
                sim_utils=sim_utils,
                camera_cfg_cls=CameraCfg,
            )
        elif self.visual_mode == "depth_rgb":
            for name, dc_cfg in build_depth_camera_cfgs(
                self._lidar_mesh_paths,
                img_res=img_res,
                update_period_s=1.0 / 15.0,
                patterns_mod=patterns,
                ray_caster_camera_cfg_cls=RayCasterCameraCfg,
            ):
                setattr(env_cfg.scene, name, dc_cfg)
                self._depth_cam_names.append(name)

        lidar_half = float(np.clip(self.lidar_h_half_fov_deg, 0.0, 180.0))
        lidar_common = dict(
            prim_path="{ENV_REGEX_NS}/Robot/head_link",
            update_period=self.lidar_period_s,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ray_alignment="base",
            pattern_cfg=patterns.LidarPatternCfg(
                channels=45,
                vertical_fov_range=(-90, 90),
                # Forward cone only — rays outside this FOV must not pull l
                # (camera cannot see the full front hemisphere).
                horizontal_fov_range=(-lidar_half, lidar_half),
                horizontal_res=2.0,
            ),
            debug_vis=False,
        )
        print(
            f"[IsaacG1Wrapper] l/h_s LiDAR cone ±{lidar_half:g}° "
            f"(outside cone → no-hit, l=2)"
        )
        for i, mesh_path in enumerate(self._lidar_mesh_paths):
            sensor_name = self._lidar_sensor_names[i]
            setattr(
                env_cfg.scene,
                sensor_name,
                RayCasterCfg(mesh_prim_paths=[mesh_path], **lidar_common),
            )

        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        self.sim_dt = float(self.env.unwrapped.cfg.sim.dt)

        # GUI + FULL_RENDERING blocks the Kit UI thread during long collect/train
        # loops → Windows "Isaac Sim (Not Responding)". Downgrade render mode.
        sim = self.env.unwrapped.sim
        if sim.has_gui():
            if self.visual_mode == "rtx_rgb":
                sim.set_render_mode(sim.RenderMode.PARTIAL_RENDERING)
                print(
                    "[WARN] GUI open: viewport set to PARTIAL_RENDERING. "
                    "For training, kill and restart with --headless."
                )
            else:
                sim.set_render_mode(sim.RenderMode.NO_RENDERING)
                print(
                    "[WARN] GUI open: viewport set to NO_RENDERING. "
                    "For training, kill and restart with --headless."
                )

        # Non-depth modes that did not pre-load lab (e.g. lidar_rgb).
        if self.visual_mode not in ("depth_rgb", "rtx_rgb"):
            load_lab_scene_usd(demos_dir=self.demos_dir)
        else:
            # G1FlatEnvCfg spawns /World/ground after any pre-env lab load.
            # Remove it again so the default grid does not z-fight the GS floor.
            stage = self._omni_usd.get_context().get_stage()
            ground_path = "/World/ground"
            if stage.GetPrimAtPath(ground_path):
                stage.RemovePrim(ground_path)
                print("[INFO] Removed default /World/ground under GS lab floor.")

        runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        runner.load(checkpoint)
        self.policy = runner.get_inference_policy(device=self.device)

        robot = self.env.unwrapped.scene["robot"]
        self.num_joints = int(robot.data.joint_pos.shape[1])
        self.proprio_dim = self.num_joints

        if self.visual_mode == "rtx_rgb":
            self.camera = self.env.unwrapped.scene["camera"]

        self._ignored_collision_links = {"left_ankle_roll_link", "right_ankle_roll_link"}
        self._setup_contact_indices()

        self.commands = torch.zeros(1, 3, device=self.device)
        self.waypoint = np.array([3.5, -0.5], dtype=np.float64)
        self.waypoints = np.array([[3.5, -0.5]], dtype=np.float64)
        self.waypoint_region_names: list[str] = []
        self.current_waypoint_idx = 0
        self._link_stuck_counters: np.ndarray | None = None
        self._active_obstacles: list[str] = []
        self._last_policy_obs = None
        self._last_lidar_stats: dict[str, float] = {
            "lidar_min_distance": float("nan"),
            "lidar_min_distance_xy": float("nan"),
            "contact_collision": 0.0,
        }

        self._rng = np.random.default_rng()
        print(
            "[IsaacG1Wrapper] layout RNG is independent of train --seed "
            "(waypoints/obstacles differ across bash runs unless --layout_seed is set)."
        )

    def _setup_contact_indices(self) -> None:
        sensor = self.env.unwrapped.scene["robot_contact"]
        link_names = sensor.body_names
        self._collision_link_indices = [
            i for i, name in enumerate(link_names) if name not in self._ignored_collision_links
        ]
        ignored = [n for n in link_names if n in self._ignored_collision_links]
        monitored = [link_names[i] for i in self._collision_link_indices]
        print(
            "[IsaacG1Wrapper] contact: ignore foot soles "
            f"{ignored}; monitor {len(monitored)} links "
            f"(thr={self.collision_force_threshold:g} N)"
        )

    def _load_scene_usd(self) -> None:
        from lab_scene_utils import load_lab_scene_usd

        load_lab_scene_usd(demos_dir=self.demos_dir, verbose=False)

    def _pose_tensor(self, pos: tuple[float, float, float], rot: tuple[float, float, float, float]) -> torch.Tensor:
        return torch.tensor(
            [[pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]]],
            device=self.device,
            dtype=torch.float32,
        )

    def _set_obstacle_pose(self, name: str, pos: tuple[float, float, float]) -> None:
        spec = self._obstacle_specs.get(name) or _bin_spec(0)
        rot = spec["rot"]
        obj = self.env.unwrapped.scene[name]
        obj.write_root_pose_to_sim(self._pose_tensor(pos, rot))
        if hasattr(obj, "write_root_velocity_to_sim"):
            obj.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=self.device, dtype=torch.float32)
            )

    def _bin_z(self, name: str) -> float:
        spec = self._obstacle_specs.get(name) or _bin_spec(0)
        return float(spec["default_z"])

    def _hide_obstacle(self, name: str) -> tuple[float, float, float]:
        pos = HIDDEN_OBSTACLE_POS
        self._set_obstacle_pose(name, pos)
        return pos

    def _choose_n_path_obstacles(self, rng: np.random.Generator) -> int:
        absent_p = float(self.obstacle_absent_prob)
        if absent_p > 0.0 and float(rng.random()) < absent_p:
            return 0
        max_n = min(2, int(self._max_n_obstacles))
        if max_n <= 1:
            return 1
        if float(rng.random()) < float(self.two_obstacle_prob):
            return 2
        return 1

    def _place_path_obstacles(
        self,
        rng: np.random.Generator,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        *,
        n_obstacles: int | None = None,
        positions: list[np.ndarray] | None = None,
    ) -> dict[str, tuple[float, float, float]]:
        """Place 0/1/2 bins on the start-goal perpendicular. Hide unused assets."""
        if positions is not None:
            xys = [np.asarray(p, dtype=np.float64).reshape(2) for p in positions]
            n = len(xys)
        else:
            n = int(self._choose_n_path_obstacles(rng) if n_obstacles is None else n_obstacles)
            n = max(0, min(n, int(self._max_n_obstacles)))
            xys = sample_perp_points_on_segment(
                rng,
                start_xy,
                goal_xy,
                n,
                perp_offset=float(self.perp_offset),
                x_min=float(self.arena_x_min),
                x_max=float(self.arena_x_max),
                y_min=float(self.arena_y_min),
                y_max=float(self.arena_y_max),
            ) if n > 0 else []
        self._n_obstacles = len(xys)
        self.obstacle_present = self._n_obstacles > 0
        self._obstacle_xy_list = [p.copy() for p in xys]
        self._active_obstacles = list(self._obstacle_names[: self._n_obstacles])
        obstacle_positions: dict[str, tuple[float, float, float]] = {}
        for i, name in enumerate(self._obstacle_names):
            z = self._bin_z(name)
            if i < self._n_obstacles:
                xy = xys[i]
                pos = (float(xy[0]), float(xy[1]), z)
                self._set_obstacle_pose(name, pos)
            else:
                pos = self._hide_obstacle(name)
            obstacle_positions[name] = pos
        if self._n_obstacles > 0:
            self._blue_bin_xy = (float(xys[0][0]), float(xys[0][1]))
        return obstacle_positions

    def _advance_collect_controller(self) -> None:
        """Alternate waypoint / switch on training resets (skip constructor reset)."""
        if not bool(getattr(self, "alternate_collect_controllers", False)):
            return
        if int(self._n_scene_resets) == 0:
            return
        if int(self._collect_ep_toggle) % 2 == 0:
            self.collect_controller = "waypoint"
        else:
            self.collect_controller = "switch"
        self._collect_ep_toggle = int(self._collect_ep_toggle) + 1

    def _sample_waypoint_sequence(self, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
        """Sample waypoint sequence from ``trajectory_region_sequence``.

        - Buffer (``include_middle_pass=True``): start → front → left|right|middle
          (cycle unless ``force_pass_side`` / ``randomize_obstacle``).
        - Formal train (``include_middle_pass=False``): start → left|right
          (switch-collect also appends behind-bin ``back``).
        - ``spawn_hemisphere_pass=True`` (formal train only): alternate spawn
          half-disk coupled to matching left/right (no front/middle).
        - ``waypoint_layout="start_goal_perp"``: start + 2 perp vias + goal.
        """
        self._spawn_yaw = None
        if str(getattr(self, "waypoint_layout", "regions")) == "start_goal_perp":
            wps, names, spawn_yaw = sample_start_goal_perp_path(
                rng,
                x_min=float(self.arena_x_min),
                x_max=float(self.arena_x_max),
                y_min=float(self.arena_y_min),
                y_max=float(self.arena_y_max),
                perp_offset=float(self.perp_offset),
                min_start_goal_dist=float(self.min_start_goal_dist),
                margin=float(self.arena_sample_margin),
            )
            self._spawn_yaw = float(spawn_yaw)
            return wps, names
        if bool(getattr(self, "spawn_hemisphere_pass", False)):
            return self._sample_hemisphere_coupled_sequence(rng)

        sequence: list[str | tuple[str, ...]] = []
        for entry in self.trajectory_region_sequence:
            if (
                self.alternate_left_right
                and isinstance(entry, tuple)
                and set(entry) >= set(PASS_SIDE_TRAIN)
            ):
                forced = getattr(self, "force_pass_side", None)
                if forced:
                    side = str(forced)
                    if side not in self.trajectory_regions:
                        raise KeyError(
                            f"force_pass_side={side!r} not in trajectory_regions "
                            f"{list(self.trajectory_regions)}"
                        )
                elif self.include_middle_pass:
                    # Critic-only: collision demos (middle) are useful; no actor to spoil.
                    if self.randomize_obstacle:
                        side = str(rng.choice(PASS_SIDE_CYCLE))
                    else:
                        side = PASS_SIDE_CYCLE[
                            self._pass_side_toggle % len(PASS_SIDE_CYCLE)
                        ]
                        self._pass_side_toggle += 1
                elif self.randomize_obstacle:
                    side = str(rng.choice(PASS_SIDE_TRAIN))
                else:
                    side = PASS_SIDE_TRAIN[self._pass_side_toggle % len(PASS_SIDE_TRAIN)]
                    self._pass_side_toggle += 1
                sequence.append(side)
            else:
                sequence.append(entry)
        return generate_random_waypoint_sequence(
            rng,
            trajectory_regions=self.trajectory_regions,
            trajectory_region_sequence=sequence,
        )

    def _sample_hemisphere_coupled_sequence(
        self, rng: np.random.Generator
    ) -> tuple[np.ndarray, list[str]]:
        """Alternate L/R spawn half-disk; pass-side region matches that half."""
        side = PASS_SIDE_TRAIN[self._hemisphere_toggle % len(PASS_SIDE_TRAIN)]
        self._hemisphere_toggle += 1
        if "start" not in self.trajectory_regions:
            raise KeyError("spawn_hemisphere_pass requires a 'start' region")
        if side not in self.trajectory_regions:
            raise KeyError(
                f"spawn_hemisphere_pass side={side!r} missing from "
                f"{list(self.trajectory_regions)}"
            )
        start_pt = sample_point_in_half_disk(
            self.trajectory_regions["start"], rng, side=side
        )
        goal_pt = sample_point_in_region(self.trajectory_regions[side], rng)
        pts = [start_pt, goal_pt]
        names = ["start", side]
        seq = list(self.trajectory_region_sequence)
        has_back = any(
            e == "back" or (isinstance(e, (tuple, list)) and "back" in e)
            for e in seq
        )
        if has_back and "back" in self.trajectory_regions:
            pts.append(sample_point_in_region(self.trajectory_regions["back"], rng))
            names.append("back")
        return np.stack(pts, axis=0), names

    def _reset_stuck_counters(self) -> None:
        n = len(self._collision_link_indices)
        self._link_stuck_counters = np.zeros(n, dtype=np.int32)

    def update_stuck_detection(self) -> bool:
        """True when any monitored link has sustained contact (stuck against obstacle)."""
        if self._link_stuck_counters is None:
            self._reset_stuck_counters()

        sensor = self.env.unwrapped.scene["robot_contact"]
        contact = sensor.data.net_forces_w[0].detach().cpu().numpy()
        for i, link_idx in enumerate(self._collision_link_indices):
            active = bool(
                np.any(np.abs(contact[link_idx, :]) > self.collision_force_threshold)
            )
            if active:
                self._link_stuck_counters[i] += 1
            else:
                self._link_stuck_counters[i] = 0

        return bool(np.any(self._link_stuck_counters >= self.stuck_contact_steps))

    def distance_to_current_waypoint(self) -> float:
        robot_xy = self.get_robot_xy_local()
        target = np.asarray(self.waypoint, dtype=np.float64).reshape(2)
        return float(np.hypot(target[0] - robot_xy[0], target[1] - robot_xy[1]))

    def _robot_xy_world(self) -> np.ndarray:
        base_pos, _ = self.get_robot_base_pose()
        return np.asarray([float(base_pos[0]), float(base_pos[1])], dtype=np.float64)

    def _passed_current_waypoint(self, robot_xy: np.ndarray) -> bool:
        """True if robot is geometrically past the current via along prev→curr.

        Needed for switching: SF may drive past a via without entering
        ``waypoint_stop_thresh``; when nominal resumes it must not turn back.
        Terminal goal is proximity-only unless ``advance_passed_terminal``.
        """
        waypoint_list = waypoints_to_list(self.waypoints)
        i = int(self.current_waypoint_idx)
        if i < 0 or i >= len(waypoint_list):
            return False
        # Last waypoint = goal: skip-by-pass only when explicitly enabled
        # (switch-collect train: left|right is the episode goal).
        if i >= len(waypoint_list) - 1 and not bool(self.advance_passed_terminal):
            return False
        curr = np.asarray(waypoint_list[i], dtype=np.float64).reshape(2)
        prev = (
            np.asarray(waypoint_list[i - 1], dtype=np.float64).reshape(2)
            if i > 0
            else curr - np.array([1.0, 0.0], dtype=np.float64)
        )
        seg = curr - prev
        seg_norm2 = float(np.dot(seg, seg))
        if seg_norm2 < 1e-8:
            # Degenerate segment: fall back to plane facing the next via,
            # or +x if this is the terminal goal.
            if i + 1 >= len(waypoint_list):
                fwd = np.array([1.0, 0.0], dtype=np.float64)
            else:
                nxt = np.asarray(waypoint_list[i + 1], dtype=np.float64).reshape(2)
                fwd = nxt - curr
            fwd_norm = float(np.linalg.norm(fwd))
            if fwd_norm < 1e-8:
                return False
            return float(np.dot(robot_xy - curr, fwd / fwd_norm)) > 0.05
        # Parametric progress along prev→curr; t>=1 means at/past curr.
        t = float(np.dot(robot_xy - prev, seg) / seg_norm2)
        return t >= 1.0

    def advance_waypoint_if_reached(self) -> bool:
        """Advance when within stop thresh OR past the via (SF overshoot).

        Returns True if all waypoints are done.
        """
        waypoint_list = waypoints_to_list(self.waypoints)
        if not waypoint_list:
            return True

        robot_xy = self.get_robot_xy_local()
        advanced = False
        while self.current_waypoint_idx < len(waypoint_list):
            dist = self.distance_to_current_waypoint()
            reached = dist < self.waypoint_stop_thresh
            passed = self._passed_current_waypoint(robot_xy)
            if not (reached or passed):
                break
            advanced = True
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(waypoint_list):
                return True
            self.waypoint = waypoint_list[self.current_waypoint_idx]
            # Loop: SF may have overshot several vias in one step.
        if advanced:
            # Refresh cached current waypoint after multi-skip.
            self.waypoint = waypoint_list[self.current_waypoint_idx]
        return False

    def reset_scene(
        self,
        seed: int | None = None,
        fixed_layout: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Reset sim, place bin, waypoints; spawn at sampled start (disk around origin).

        Obstacle: fixed at ``_blue_bin_xy_fixed`` when ``randomize_obstacle=False``.
        When True, resample y on bin x within ``obstacle_y_range``.
        Independently, with ``obstacle_absent_prob`` park the bin behind the
        robot spawn (facing +x → behind = smaller x) so front camera/LiDAR
        cannot see it; LiDAR hits are also ignored so l stays clean.

        If ``fixed_layout`` is provided (from a prior reset return / capture),
        reuse that bin pose + waypoint sequence instead of resampling — used by
        multi-mode eval so SF / waypoint / switching see the same scene.
        """
        if seed is not None:
            # Path layout: do not reseed from yaml train seed=0, or every
            # bash run repeats the same buffer trajectories.
            use_path = bool(getattr(self, "path_obstacle_layout", False)) or (
                str(getattr(self, "waypoint_layout", "regions")) == "start_goal_perp"
            )
            if not use_path:
                self._rng = np.random.default_rng(seed)

        self._advance_collect_controller()
        self._n_scene_resets = int(self._n_scene_resets) + 1
        self._sidestep_xy = None
        self._sidestep_hold = 0

        obs, _ = self.env.reset()

        use_path_obs = bool(getattr(self, "path_obstacle_layout", False)) or (
            str(getattr(self, "waypoint_layout", "regions")) == "start_goal_perp"
        )

        if fixed_layout is not None:
            self.waypoints = np.asarray(fixed_layout["waypoints"], dtype=np.float64).copy()
            self.waypoint_region_names = list(
                fixed_layout.get("waypoint_region_names", [])
            )
            if "spawn_yaw" in fixed_layout:
                self._spawn_yaw = float(fixed_layout["spawn_yaw"])
            if use_path_obs:
                raw_xy = fixed_layout.get("obstacle_xys")
                if raw_xy is None:
                    pos_map = fixed_layout.get("obstacle_positions") or {}
                    raw_xy = [
                        np.array([float(v[0]), float(v[1])], dtype=np.float64)
                        for _, v in sorted(pos_map.items())
                        if abs(float(v[0]) - HIDDEN_OBSTACLE_POS[0]) > 10.0
                    ]
                n_obs = int(fixed_layout.get("n_obstacles", len(raw_xy)))
                self.obstacle_present = bool(fixed_layout.get("obstacle_present", n_obs > 0))
            else:
                bin_xy = fixed_layout.get("blue_bin_xy", fixed_layout.get("bin_xy"))
                if bin_xy is None:
                    raise ValueError("fixed_layout requires blue_bin_xy")
                self._blue_bin_xy = (float(bin_xy[0]), float(bin_xy[1]))
                self.obstacle_present = bool(fixed_layout.get("obstacle_present", True))
        else:
            self.waypoints, self.waypoint_region_names = self._sample_waypoint_sequence(
                self._rng
            )
            if not use_path_obs:
                bin_x = float(self._blue_bin_xy_fixed[0])
                if self.randomize_obstacle:
                    y0, y1 = self.obstacle_y_range
                    bin_y = float(self._rng.uniform(float(y0), float(y1)))
                else:
                    bin_y = float(self._blue_bin_xy_fixed[1])
                self._blue_bin_xy = (bin_x, bin_y)
                absent_p = float(self.obstacle_absent_prob)
                self.obstacle_present = bool(
                    absent_p <= 0.0 or self._rng.random() >= absent_p
                )

        waypoint_list = waypoints_to_list(self.waypoints)
        start_xy = waypoint_list[0]
        goal_xy = waypoint_list[-1]

        obstacle_positions: dict[str, tuple[float, float, float]] = {}
        if use_path_obs:
            extra_xy = None
            extra_n = None
            if fixed_layout is not None:
                extra_xy = [
                    np.asarray(p, dtype=np.float64).reshape(2)
                    for p in (fixed_layout.get("obstacle_xys") or [])
                ]
                extra_n = int(fixed_layout.get("n_obstacles", len(extra_xy)))
                if extra_n == 0:
                    extra_xy = []
            obstacle_positions = self._place_path_obstacles(
                self._rng,
                start_xy,
                goal_xy,
                n_obstacles=extra_n,
                positions=extra_xy,
            )
        elif self.obstacle_present:
            self._active_obstacles = list(self._obstacle_names)
            self._n_obstacles = len(self._active_obstacles)
            for name in self._obstacle_names:
                spec = self._obstacle_specs[name]
                if name == "blue_bin_0":
                    x, y = self._blue_bin_xy
                else:
                    x, y = spec["spawn_xy"]
                z = float(spec["default_z"])
                self._set_obstacle_pose(name, (x, y, z))
                obstacle_positions[name] = (x, y, z)
        else:
            self._active_obstacles = []
            self._n_obstacles = 0
            behind_xy = (
                float(start_xy[0]) - 2.5,
                float(start_xy[1]),
            )
            for name in self._obstacle_names:
                z = self._bin_z(name)
                pos = (behind_xy[0], behind_xy[1], z)
                self._set_obstacle_pose(name, pos)
                obstacle_positions[name] = pos

        waypoint_list = waypoints_to_list(self.waypoints)
        self.current_waypoint_idx = 0
        self.waypoint = waypoint_list[0].copy()
        robot_xy = waypoint_list[0].copy()
        robot = self.env.unwrapped.scene["robot"]
        spawn_yaw = getattr(self, "_spawn_yaw", None)
        if spawn_yaw is None:
            spawn_quat = (1.0, 0.0, 0.0, 0.0)
        else:
            half = 0.5 * float(spawn_yaw)
            spawn_quat = (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))
        root_pose = self._pose_tensor(
            (float(robot_xy[0]), float(robot_xy[1]), 0.8),
            spawn_quat,
        )
        robot.write_root_pose_to_sim(root_pose)
        if hasattr(robot, "write_root_velocity_to_sim"):
            robot.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=self.device, dtype=torch.float32)
            )
        self._reset_stuck_counters()

        self.commands.zero_()

        # Warm up physics/sensors after teleporting assets.
        # RTX CameraCfg update_period is ~1/15s; with sim_dt=0.005 that needs
        # ~14+ steps before annotator RGB is valid. Too few steps → overscan
        # NoneType crash in omni.replicator (_resize_data_for_overscan).
        zero_actions = torch.zeros(1, self.num_joints, device=self.device)
        warmup_steps = 3
        if self.visual_mode == "rtx_rgb":
            cam_period = float(
                getattr(self.camera.cfg, "update_period", 1.0 / 15.0) or (1.0 / 15.0)
            )
            warmup_steps = max(30, int(np.ceil(cam_period / max(self.sim_dt, 1e-6))) + 5)
        for _ in range(warmup_steps):
            self.env.unwrapped.command_manager._terms["base_velocity"].command[:] = self.commands
            obs, _, _, _ = self.env.step(zero_actions)

        # Warp LiDAR meshes are static at spawn. Rebake after bins have been
        # teleported and PhysX/USD have been stepped, otherwise dual-bin l
        # reports range to the old mesh (often the farther bin).
        self._refresh_lidar_warp_meshes()
        self._last_policy_obs = obs
        self._update_lidar_stats()

        return {
            "active_obstacles": list(self._active_obstacles),
            "obstacle_present": bool(self.obstacle_present),
            "n_obstacles": int(getattr(self, "_n_obstacles", len(self._active_obstacles))),
            "robot_xy": robot_xy,
            "blue_bin_xy": (float(self._blue_bin_xy[0]), float(self._blue_bin_xy[1])),
            "obstacle_xys": [
                (float(p[0]), float(p[1]))
                for p in getattr(self, "_obstacle_xy_list", [])
            ],
            "obstacle_positions": obstacle_positions,
            "collect_controller": str(getattr(self, "collect_controller", "sf")),
            "waypoint": self.waypoint.copy(),
            "waypoints": self.waypoints.copy(),
            "waypoint_region_names": list(self.waypoint_region_names),
            "current_waypoint_idx": self.current_waypoint_idx,
            **self._last_lidar_stats,
        }

    def _rebake_raycaster_warp_mesh(self, lidar) -> None:
        """Rebuild Warp mesh at the obstacle's *current* world pose.

        Isaac Lab RayCaster only supports static meshes: vertices are baked in
        world frame at sensor init. Moving a rigid bin with write_root_pose
        does not move the raycast mesh, so dual-bin path layout would report
        range to the spawn/ghost mesh (often the other bin) instead of the
        closest visual bin.
        """
        import isaaclab.sim as sim_utils
        import omni.usd
        from isaaclab.utils.warp import convert_to_warp_mesh
        from pxr import UsdGeom

        mesh_prim_path = lidar.cfg.mesh_prim_paths[0]
        mesh_prim = sim_utils.get_first_matching_child_prim(
            mesh_prim_path, lambda prim: prim.GetTypeName() == "Mesh"
        )
        if mesh_prim is None or not mesh_prim.IsValid():
            print(f"[WARN] lidar rebake: no Mesh under {mesh_prim_path}")
            return
        mesh_prim = UsdGeom.Mesh(mesh_prim)
        points = np.asarray(mesh_prim.GetPointsAttr().Get())
        transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
        points = np.matmul(points, transform_matrix[:3, :3].T)
        points += transform_matrix[:3, 3]
        indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
        lidar.meshes[mesh_prim_path] = convert_to_warp_mesh(
            points, indices, device=self.device
        )

    def _refresh_lidar_warp_meshes(self) -> None:
        scene = self.env.unwrapped.scene
        n = 0
        # InteractiveScene has no __contains__; `name not in scene` probes
        # scene[0] and crashes with KeyError('0').
        for name in list(getattr(self, "_lidar_sensor_names", []) or []):
            try:
                lidar = scene[str(name)]
            except KeyError:
                continue
            try:
                self._rebake_raycaster_warp_mesh(lidar)
                n += 1
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] lidar rebake failed for {name}: {exc}")
        if n and int(getattr(self, "_n_scene_resets", 0)) <= 2:
            print(
                f"[IsaacG1Wrapper] rebaked {n} RayCaster warp mesh(es) "
                "to follow moved bins (static LiDAR meshes)."
            )

    def _forward_cone_cos_min(self) -> float:
        half_deg = float(np.clip(getattr(self, "lidar_h_half_fov_deg", 90.0), 0.0, 180.0))
        return float(np.cos(np.deg2rad(half_deg)))

    def _xy_in_forward_cone(
        self, dx: np.ndarray | float, dy: np.ndarray | float, c: float, s: float
    ) -> np.ndarray:
        """True where (dx, dy) lies in the heading-aligned LiDAR cone."""
        dx_a = np.asarray(dx, dtype=np.float64)
        dy_a = np.asarray(dy, dtype=np.float64)
        # Missed rays are inf; inf*c + inf*s is NaN when yaw cos/sin have
        # opposite signs. Treat non-finite hits as outside the cone.
        finite = np.isfinite(dx_a) & np.isfinite(dy_a)
        dx_s = np.where(finite, dx_a, 0.0)
        dy_s = np.where(finite, dy_a, 0.0)
        d = np.hypot(dx_s, dy_s)
        fwd = dx_s * c + dy_s * s
        return finite & (d > 1e-8) & (fwd >= d * self._forward_cone_cos_min() - 1e-6)

    def _front_obstacle_xy_min(self) -> float:
        """Min XY range to active bins inside the forward LiDAR cone."""
        xys = getattr(self, "_obstacle_xy_list", None) or []
        if not xys:
            return float("nan")
        robot_xy = self.get_robot_xy_local()
        _, quat = self.get_robot_base_pose()
        yaw = quat_to_yaw(np.asarray(quat, dtype=np.float64).reshape(-1))
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        best = float("inf")
        for p in xys:
            dx = float(p[0]) - float(robot_xy[0])
            dy = float(p[1]) - float(robot_xy[1])
            if not bool(self._xy_in_forward_cone(dx, dy, c, s)):
                continue
            d = float(np.hypot(dx, dy))
            if d < best:
                best = d
        return best if np.isfinite(best) else float("nan")

    def get_lidar_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        # Despawned obstacle: ignore mesh hits entirely so l is not polluted.
        if not bool(getattr(self, "obstacle_present", True)):
            stats = {
                "lidar_min_distance": float("nan"),
                "lidar_min_distance_xy": float("nan"),
            }
            empty = np.zeros((0, 3), dtype=np.float64)
            empty_r = np.zeros((0,), dtype=np.float64)
            return empty, empty_r, empty_r, stats

        scene = self.env.unwrapped.scene
        sensor_names = list(self._lidar_sensor_names)
        if bool(getattr(self, "filter_lidar_to_active_obstacles", False)):
            mapped = [
                self._obstacle_lidar_names[name]
                for name in self._active_obstacles
                if name in getattr(self, "_obstacle_lidar_names", {})
            ]
            if not mapped:
                stats = {
                    "lidar_min_distance": float("nan"),
                    "lidar_min_distance_xy": float("nan"),
                }
                empty = np.zeros((0, 3), dtype=np.float64)
                empty_r = np.zeros((0,), dtype=np.float64)
                return empty, empty_r, empty_r, stats
            sensor_names = mapped
        lidars = [scene[name] for name in sensor_names]
        hits_list = [lidar.data.ray_hits_w[0].detach().cpu().numpy() for lidar in lidars]
        origin = lidars[0].data.pos_w[0].detach().cpu().numpy()
        max_d = float(getattr(lidars[0].cfg, "max_distance", 1e6))
        diff_w, ranges, ranges_xy = merge_ray_hits_multi(origin, hits_list, max_d)
        valid_ray = np.isfinite(diff_w).all(axis=1) & (ranges > 1e-4) & (ranges < max_d * 0.999)
        _, quat = self.get_robot_base_pose()
        yaw = quat_to_yaw(np.asarray(quat, dtype=np.float64).reshape(-1))
        c, s = float(np.cos(yaw)), float(np.sin(yaw))
        valid_ray = valid_ray & self._xy_in_forward_cone(
            diff_w[:, 0], diff_w[:, 1], c, s
        )
        positive_ranges = ranges[valid_ray]
        positive_ranges_xy = ranges_xy[valid_ray]
        lidar_min_m = (
            float(np.min(positive_ranges)) if positive_ranges.size > 0 else float("nan")
        )
        lidar_min_xy_m = (
            float(np.min(positive_ranges_xy)) if positive_ranges_xy.size > 0 else float("nan")
        )
        geom_xy = self._front_obstacle_xy_min()
        if np.isfinite(geom_xy):
            lidar_min_m = (
                geom_xy if not np.isfinite(lidar_min_m) else min(lidar_min_m, geom_xy)
            )
            lidar_min_xy_m = (
                geom_xy
                if not np.isfinite(lidar_min_xy_m)
                else min(lidar_min_xy_m, geom_xy)
            )

        stats = {
            "lidar_min_distance": lidar_min_m,
            "lidar_min_distance_xy": lidar_min_xy_m,
            "lidar_geom_xy": float(geom_xy) if np.isfinite(geom_xy) else float("nan"),
        }
        return diff_w, ranges, ranges_xy, stats

    def _update_lidar_stats(self) -> None:
        _, _, _, stats = self.get_lidar_data()
        self._last_lidar_stats.update(stats)

    def get_contact_collision(self) -> float:
        sensor = self.env.unwrapped.scene["robot_contact"]
        contact = sensor.data.net_forces_w[0].detach().cpu().numpy()
        is_collision = float(
            np.any(
                np.abs(contact[self._collision_link_indices, :]) > self.collision_force_threshold
            )
        )
        self._last_lidar_stats["contact_collision"] = is_collision
        return is_collision

    def get_safety_diagnostics(self) -> dict[str, float]:
        """Return LiDAR + contact fields for validation logging.

        ``h_s`` is the continuous PyHJ avoid cost:
        ``lidar_min_distance - lidar_distance_threshold`` (<0 unsafe, >0 safe).
        If no valid hit in the forward cone (``lidar_h_half_fov_deg``),
        set ``h_s = 2.0``.
        If ``include_contact_in_hs`` and a non-foot link hits (force above
        threshold), ``h_s = min(h_s, contact_hs)`` so contact is a failure
        label. Foot soles ``left/right_ankle_roll_link`` are ignored.
        """
        self._update_lidar_stats()
        contact = self.get_contact_collision()
        lidar_dist = self._last_lidar_stats.get("lidar_min_distance", float("nan"))
        lidar_xy = self._last_lidar_stats.get("lidar_min_distance_xy", float("nan"))
        lidar_unsafe = float(
            np.isfinite(lidar_dist) and lidar_dist < self.lidar_distance_threshold
        )
        if np.isfinite(lidar_dist):
            h_s = float(lidar_dist - self.lidar_distance_threshold)
        else:
            # Forward cone clear / no hit: fixed safe margin for HJ (not NaN).
            h_s = 2.0
        if bool(getattr(self, "include_contact_in_hs", False)) and float(contact) > 0.5:
            h_s = min(h_s, float(self.contact_hs))
        return {
            "lidar_min_distance": float(lidar_dist),
            "lidar_min_distance_xy": float(lidar_xy),
            "contact_collision": float(contact),
            "lidar_unsafe": lidar_unsafe,
            "h_s": h_s,
        }

    def calculate_cost(self) -> float:
        """Continuous safety cost: dist - threshold (<0 unsafe, >0 safe)."""
        return float(self.get_safety_diagnostics()["h_s"])


    def get_lidar_min_distance(self) -> float:
        self._update_lidar_stats()
        return float(self._last_lidar_stats["lidar_min_distance"])

    def _read_rtx_rgb_hwc(self) -> np.ndarray:
        """Read RTX camera RGB; step+retry if annotator overscan is not ready yet."""
        last_exc: Exception | None = None
        zero_actions = torch.zeros(1, self.num_joints, device=self.device)
        for attempt in range(15):
            try:
                rgb_tensor = self.camera.data.output["rgb"][0]
                rgb_np = rgb_tensor[..., :3].detach().cpu().numpy()
                if rgb_np.dtype != np.uint8:
                    rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)
                return self._rotate_sensor_ccw_to_landscape(rgb_np)
            except TypeError as exc:
                # omni.replicator: datawindow_overscan_* is None on first bad frames.
                last_exc = exc
                self.env.unwrapped.command_manager._terms["base_velocity"].command[:] = (
                    self.commands
                )
                obs, _, _, _ = self.env.step(zero_actions)
                self._last_policy_obs = obs
                if attempt == 0 or attempt == 14:
                    print(
                        f"[WARN] rtx_rgb read failed (attempt {attempt + 1}/15): {exc}"
                    )
        raise RuntimeError(
            "rtx_rgb camera never produced a valid RGB frame. "
            "Try without --headless, or use --visual_mode depth_rgb."
        ) from last_exc

    def get_raw_obs(self) -> dict[str, np.ndarray]:
        if not self.collect_visual:
            raise RuntimeError(
                f"visual_mode={self.visual_mode!r} — set --visual_mode depth_rgb (default on cluster)."
            )

        if self.visual_mode == "rtx_rgb":
            visual = self._read_rtx_rgb_hwc()
        elif self.visual_mode == "depth_rgb":
            scene = self.env.unwrapped.scene
            depth_list = [
                scene[n].data.output["distance_to_image_plane"][0].detach().cpu().numpy()
                for n in self._depth_cam_names
            ]
            merged = self._merge_depth_maps_multi(depth_list, max_d=10.0)
            visual = self._rotate_sensor_ccw_to_landscape(self._depth_to_rgb(merged))
        elif self.visual_mode == "lidar_rgb":
            _, ranges, _, _ = self.get_lidar_data()
            valid = np.isfinite(ranges) & (ranges > 1e-4) & (ranges < 10.0)
            visual = self._rotate_sensor_ccw_to_landscape(
                self._lidar_ranges_to_rgb(ranges, valid, out_size=self.img_res)
            )
        else:
            raise RuntimeError(f"Unsupported visual_mode: {self.visual_mode}")

        robot = self.env.unwrapped.scene["robot"]
        proprio = robot.data.joint_pos[0, : self.num_joints].detach().cpu().numpy().astype(np.float32)

        return {"visual": visual, "proprio": proprio}

    def get_robot_base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (base_pos_w xyz, base_quat_w wxyz) as float64 numpy arrays."""
        robot = self.env.unwrapped.scene["robot"]
        data = robot.data
        base_pos = data.root_pos_w[0].detach().cpu().numpy().astype(np.float64)
        base_quat = data.root_quat_w[0].detach().cpu().numpy().astype(np.float64)
        return base_pos, base_quat

    def get_robot_xy_local(self) -> np.ndarray:
        """Robot XY relative to env origin (matches waypoint / bin coords)."""
        base_pos, _ = self.get_robot_base_pose()
        origin = self.env.unwrapped.scene.env_origins[0].detach().cpu().numpy()
        return np.asarray(
            [float(base_pos[0] - origin[0]), float(base_pos[1] - origin[1])],
            dtype=np.float64,
        )

    def perp_sidestep_xy(self) -> np.ndarray | None:
        """Fresh G: right triangle robot–nearest-bin–G, right angle at bin."""
        xys = list(getattr(self, "_obstacle_xy_list", []) or [])
        if not xys:
            return None
        robot_xy = self.get_robot_xy_local()
        _, quat = self.get_robot_base_pose()
        yaw = quat_to_yaw(np.asarray(quat, dtype=np.float64).reshape(-1))
        kwargs = {
            "radius": float(
                getattr(self, "lidar_distance_threshold", DEFAULT_DANGER_RADIUS_M)
            ),
            "heading": yaw,
        }
        if bool(getattr(self, "use_arena_bounds", False)):
            kwargs.update(
                x_min=float(self.arena_x_min),
                x_max=float(self.arena_x_max),
                y_min=float(self.arena_y_min),
                y_max=float(self.arena_y_max),
            )
        return compute_perp_sidestep_xy(robot_xy, xys, **kwargs)

    def update_sidestep_cache(self, use_sf: bool) -> bool:
        """Hold G for ``sidestep_hold_steps`` consecutive HJ<0 steps.

        First HJ<0 computes G. Recompute only after that many steps if still
        HJ<0. Leaving HJ<0 clears the cache. Returns True if G was (re)computed.
        """
        if not use_sf:
            self._sidestep_xy = None
            self._sidestep_hold = 0
            return False
        hold_n = max(1, int(getattr(self, "sidestep_hold_steps", 20)))
        refreshed = False
        if self._sidestep_xy is None or int(self._sidestep_hold) >= hold_n:
            self._sidestep_xy = self.perp_sidestep_xy()
            self._sidestep_hold = 0
            refreshed = True
        self._sidestep_hold = int(self._sidestep_hold) + 1
        return refreshed

    def cached_sidestep_xy(self) -> np.ndarray | None:
        xy = getattr(self, "_sidestep_xy", None)
        if xy is None:
            return None
        return np.asarray(xy, dtype=np.float64).reshape(2)

    def is_out_of_y_bounds(self) -> bool:
        """True if |y - y_center| exceeds soft corridor. Disabled when y_bound <= 0."""
        if self.y_bound <= 0.0:
            return False
        xy = self.get_robot_xy_local()
        return bool(abs(float(xy[1]) - float(self.y_center)) > self.y_bound)

    def is_out_of_x_bounds(self) -> bool:
        """True if env-local x exceeds the far boundary (default x >= 4)."""
        xy = self.get_robot_xy_local()
        return bool(float(xy[0]) >= self.x_bound_max)

    def is_out_of_arena(self) -> bool:
        """True if XY is on/outside the closed rectangle [x_min,x_max]×[y_min,y_max]."""
        xy = self.get_robot_xy_local()
        x, y = float(xy[0]), float(xy[1])
        return bool(
            x <= float(self.arena_x_min)
            or x >= float(self.arena_x_max)
            or y <= float(self.arena_y_min)
            or y >= float(self.arena_y_max)
        )

    def is_out_of_bounds(self) -> bool:
        """Soft bounds: arena rectangle, or original |y| corridor / x far wall."""
        if bool(getattr(self, "use_arena_bounds", False)):
            return self.is_out_of_arena()
        return self.is_out_of_y_bounds() or self.is_out_of_x_bounds()

    def get_full_state(self) -> np.ndarray:
        """Flat state vector for offline dataset storage."""
        robot = self.env.unwrapped.scene["robot"]
        data = robot.data
        base_pos = data.root_pos_w[0].cpu().numpy()
        base_quat = data.root_quat_w[0].cpu().numpy()
        joint_pos = data.joint_pos[0, : self.num_joints].cpu().numpy()
        joint_vel = data.joint_vel[0, : self.num_joints].cpu().numpy()
        cmds = self.commands[0].detach().cpu().numpy()
        # Fresh read; same label as DataCollection_test ``lidar_min_range_nonzero_m``.
        _, _, _, lidar_stats = self.get_lidar_data()
        lidar_min = np.array([lidar_stats["lidar_min_distance"]], dtype=np.float64)
        return np.concatenate([base_pos, base_quat, joint_pos, joint_vel, cmds, lidar_min])

    def apply_velocity_command(self, action: np.ndarray | torch.Tensor) -> tuple[Any, bool, bool, dict]:
        """Apply (vx, vy, yaw_rate), run PPO, return (policy_obs, terminated, truncated, info)."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 3:
            raise ValueError(f"Expected 3-d velocity action, got shape {action.shape}")

        # HJ / env action: default vx in [0, 0.8]; vy in [-0.5, 0.5]; yaw in [-0.5, 0.5].
        # Path pipeline may widen yaw via action_low/action_high.
        low = np.asarray(self.action_low, dtype=np.float32).reshape(-1)
        high = np.asarray(self.action_high, dtype=np.float32).reshape(-1)
        vx = float(np.clip(action[0], float(low[0]), float(high[0])))
        vy = float(np.clip(action[1], float(low[1]), float(high[1])))
        yaw_rate = float(np.clip(action[2], float(low[2]), float(high[2])))

        self.commands[0, 0] = vx
        self.commands[0, 1] = vy
        self.commands[0, 2] = yaw_rate

        self.env.unwrapped.command_manager._terms["base_velocity"].command[:] = self.commands

        obs = self._last_policy_obs
        if obs is None:
            obs, _ = self.env.reset()

        obs["policy"][0, 11] = self.commands[0, 2]

        with torch.inference_mode():
            actions = self.policy(obs)

        obs, rew, dones, extras = self.env.step(actions)
        self._last_policy_obs = obs
        self._update_lidar_stats()
        self.get_contact_collision()
        stuck = self.update_stuck_detection()

        info = dict(extras) if extras is not None else {}
        info.update(self._last_lidar_stats)
        info["waypoint"] = self.waypoint.copy()
        info["velocity_command"] = np.array([vx, vy, yaw_rate], dtype=np.float32)
        info["stuck"] = stuck

        if isinstance(dones, torch.Tensor):
            done = bool(dones[0].item())
        else:
            done = bool(dones)
        return obs, done, False, info

    def close(self) -> None:
        pass
