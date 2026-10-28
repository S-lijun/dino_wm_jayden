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
    DEFAULT_TRAJECTORY_REGIONS,
    DEFAULT_TRAJECTORY_REGION_SEQUENCE,
    generate_random_waypoint_sequence,
    waypoints_to_list,
)

# Obstacle scene keys registered on env_cfg.scene (see data_collection_obstacles.py).
OBSTACLE_SPECS: dict[str, dict[str, Any]] = {
    "blue_bin_0": {
        "default_z": 0.5,
        "rot": (0.5, 0.5, 0.5, 0.5),
        "spawn_xy": (2.0, 0.0),
    },
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
        lidar_distance_threshold: float = 0.3,
        collision_force_threshold: float = 0.1,
        stuck_contact_steps: int = 50,
        waypoint_stop_thresh: float = 0.1,
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
        self.env_prim_root = env_prim_root
        self.lidar_distance_threshold = lidar_distance_threshold
        self.collision_force_threshold = collision_force_threshold
        self.stuck_contact_steps = int(stuck_contact_steps)
        self.waypoint_stop_thresh = float(waypoint_stop_thresh)
        self.trajectory_regions = (
            trajectory_regions if trajectory_regions is not None else DEFAULT_TRAJECTORY_REGIONS
        )
        self.trajectory_region_sequence = (
            trajectory_region_sequence
            if trajectory_region_sequence is not None
            else DEFAULT_TRAJECTORY_REGION_SEQUENCE
        )
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

        self._obstacle_names = list(OBSTACLE_SPECS.keys())
        self._depth_cam_names: list[str] = []

        if self.visual_mode == "depth_rgb":
            load_lab_scene_usd(demos_dir=self.demos_dir)
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

        add_blue_bin(env_cfg, pos=(2.0, 0.0, 0.5), index=0)

        env_cfg.scene.robot_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*link.*",
            update_period=0.0,
            debug_vis=False,
            filter_prim_paths_expr=[],
        )

        self.lidar_fps = 7.0
        self.lidar_period_s = 1.0 / self.lidar_fps

        if self.visual_mode == "rtx_rgb":
            env_cfg.scene.camera = build_rtx_camera_cfg(
                img_res=img_res,
                update_period_s=1.0 / 15.0,
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

        lidar_common = dict(
            prim_path="{ENV_REGEX_NS}/Robot/head_link",
            update_period=self.lidar_period_s,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
            ray_alignment="base",
            pattern_cfg=patterns.LidarPatternCfg(
                channels=45,
                vertical_fov_range=(-90, 90),
                horizontal_fov_range=(-180, 180),
                horizontal_res=2.0,
            ),
            debug_vis=False,
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

        if self.visual_mode != "depth_rgb":
            load_lab_scene_usd(demos_dir=self.demos_dir)

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
        self.waypoint = np.array([2.0, 1.0], dtype=np.float64)
        self.waypoints = np.array([[2.0, 1.0]], dtype=np.float64)
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

    def _setup_contact_indices(self) -> None:
        sensor = self.env.unwrapped.scene["robot_contact"]
        link_names = sensor.body_names
        self._collision_link_indices = [
            i for i, name in enumerate(link_names) if name not in self._ignored_collision_links
        ]

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
        rot = OBSTACLE_SPECS[name]["rot"]
        obj = self.env.unwrapped.scene[name]
        obj.write_root_pose_to_sim(self._pose_tensor(pos, rot))
        if hasattr(obj, "write_root_velocity_to_sim"):
            obj.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=self.device, dtype=torch.float32)
            )

    def _sample_waypoint_sequence(self, rng: np.random.Generator) -> tuple[np.ndarray, list[str]]:
        return generate_random_waypoint_sequence(
            rng,
            trajectory_regions=self.trajectory_regions,
            trajectory_region_sequence=self.trajectory_region_sequence,
        )

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
        robot = self.env.unwrapped.scene["robot"]
        base_pos = robot.data.root_pos_w[0].cpu().numpy()
        target = self.waypoint
        return float(np.hypot(target[0] - base_pos[0], target[1] - base_pos[1]))

    def advance_waypoint_if_reached(self) -> bool:
        """Move to next region waypoint when within stop threshold. Returns True if all done."""
        if self.distance_to_current_waypoint() >= self.waypoint_stop_thresh:
            return False

        self.current_waypoint_idx += 1
        waypoint_list = waypoints_to_list(self.waypoints)
        if self.current_waypoint_idx >= len(waypoint_list):
            return True

        self.waypoint = waypoint_list[self.current_waypoint_idx]
        return False

    def reset_scene(self, seed: int | None = None) -> dict[str, Any]:
        """Reset sim, place blue_bin obstacle, region waypoints, spawn at first waypoint."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        obs, _ = self.env.reset()

        self.waypoints, self.waypoint_region_names = self._sample_waypoint_sequence(self._rng)
        waypoint_list = waypoints_to_list(self.waypoints)
        self.current_waypoint_idx = 0
        self.waypoint = waypoint_list[0].copy()
        start_xy = waypoint_list[0]

        robot = self.env.unwrapped.scene["robot"]
        robot_xy = start_xy.copy()
        root_pose = self._pose_tensor(
            (float(robot_xy[0]), float(robot_xy[1]), 0.8),
            (1.0, 0.0, 0.0, 0.0),
        )
        robot.write_root_pose_to_sim(root_pose)
        if hasattr(robot, "write_root_velocity_to_sim"):
            robot.write_root_velocity_to_sim(
                torch.zeros(1, 6, device=self.device, dtype=torch.float32)
            )
        self._reset_stuck_counters()

        self._active_obstacles = list(self._obstacle_names)
        obstacle_positions: dict[str, tuple[float, float, float]] = {}

        for name in self._obstacle_names:
            if name == "blue_bin_0":
                x, y = self._blue_bin_xy
            else:
                x, y = OBSTACLE_SPECS[name]["spawn_xy"]
            z = OBSTACLE_SPECS[name]["default_z"]
            self._set_obstacle_pose(name, (x, y, z))
            obstacle_positions[name] = (x, y, z)

        self.commands.zero_()

        # Warm up physics/sensors after teleporting assets.
        zero_actions = torch.zeros(1, self.num_joints, device=self.device)
        for _ in range(3):
            self.env.unwrapped.command_manager._terms["base_velocity"].command[:] = self.commands
            obs, _, _, _ = self.env.step(zero_actions)

        self._last_policy_obs = obs
        self._update_lidar_stats()

        return {
            "active_obstacles": list(self._active_obstacles),
            "robot_xy": robot_xy,
            "obstacle_positions": obstacle_positions,
            "waypoint": self.waypoint.copy(),
            "waypoints": self.waypoints.copy(),
            "waypoint_region_names": list(self.waypoint_region_names),
            "current_waypoint_idx": self.current_waypoint_idx,
            **self._last_lidar_stats,
        }

    def get_lidar_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
        scene = self.env.unwrapped.scene
        lidars = [scene[name] for name in self._lidar_sensor_names]
        hits_list = [lidar.data.ray_hits_w[0].detach().cpu().numpy() for lidar in lidars]
        origin = lidars[0].data.pos_w[0].detach().cpu().numpy()
        max_d = float(getattr(lidars[0].cfg, "max_distance", 1e6))
        diff_w, ranges, ranges_xy = merge_ray_hits_multi(origin, hits_list, max_d)
        valid_ray = np.isfinite(diff_w).all(axis=1) & (ranges > 1e-4) & (ranges < max_d * 0.999)
        positive_ranges = ranges[valid_ray]
        positive_ranges_xy = ranges_xy[valid_ray]
        lidar_min_m = (
            float(np.min(positive_ranges)) if positive_ranges.size > 0 else float("nan")
        )
        lidar_min_xy_m = (
            float(np.min(positive_ranges_xy)) if positive_ranges_xy.size > 0 else float("nan")
        )

        stats = {
            "lidar_min_distance": lidar_min_m,
            "lidar_min_distance_xy": lidar_min_xy_m,
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
        """Return LiDAR + contact fields for validation logging."""
        self._update_lidar_stats()
        contact = self.get_contact_collision()
        lidar_dist = self._last_lidar_stats.get("lidar_min_distance", float("nan"))
        lidar_xy = self._last_lidar_stats.get("lidar_min_distance_xy", float("nan"))
        lidar_unsafe = float(
            np.isfinite(lidar_dist) and lidar_dist < self.lidar_distance_threshold
        )
        h_s = 1.0 if (contact > 0.5 or lidar_unsafe > 0.5) else 0.0
        return {
            "lidar_min_distance": float(lidar_dist),
            "lidar_min_distance_xy": float(lidar_xy),
            "contact_collision": float(contact),
            "lidar_unsafe": lidar_unsafe,
            "h_s": h_s,
        }

    def calculate_cost(self) -> float:
        """Safety cost: 1.0 if contact collision or LiDAR too close, else 0.0."""
        return float(self.get_safety_diagnostics()["h_s"])


    def get_lidar_min_distance(self) -> float:
        self._update_lidar_stats()
        return float(self._last_lidar_stats["lidar_min_distance"])

    def get_raw_obs(self) -> dict[str, np.ndarray]:
        if not self.collect_visual:
            raise RuntimeError(
                f"visual_mode={self.visual_mode!r} — set --visual_mode depth_rgb (default on cluster)."
            )

        if self.visual_mode == "rtx_rgb":
            rgb_tensor = self.camera.data.output["rgb"][0]
            rgb_np = rgb_tensor[..., :3].detach().cpu().numpy()
            if rgb_np.dtype != np.uint8:
                rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)
            visual = self._rotate_sensor_ccw_to_landscape(rgb_np)
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

        vx = float(np.clip(action[0], -self.max_speed, self.max_speed))
        vy = float(np.clip(action[1], -self.max_speed, self.max_speed))
        yaw_rate = float(np.clip(action[2], -1.0, 1.0))

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
