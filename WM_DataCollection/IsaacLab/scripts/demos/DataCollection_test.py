
"""
Collect G1 locomotion data (same as DataCollection_test.py) with multi RayCasters
(lidar_0 / lidar_1 / ...), each targeting one mesh prim. Per-ray hits are merged by
taking the closest valid intersection so CSV / npy match the single-lidar script's shape.
"""

import os
import csv
import sys
import torch
import numpy as np
import argparse
from datetime import datetime

# ---------------------------------------------------------------------
# Isaac Lab launcher
# ---------------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Collect G1 locomotion data with multi LiDAR (one mesh per lidar, merged per ray)."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--no_collect", action="store_true", help="Disable data collection (only visualize robot).")
parser.add_argument("--waypoint_x", type=float, default=2.0, help="Waypoint X position in world coordinates.")
parser.add_argument("--waypoint_y", type=float, default=1.0, help="Waypoint Y position in world coordinates.")
parser.add_argument("--vx", type=float, default=0.5, help="Initial vx command (m/s).")
parser.add_argument("--vy", type=float, default=0.0, help="Initial vy command (m/s).")
parser.add_argument("--yaw_rate", type=float, default=0.0, help="Initial yaw rate (rad/s).")
parser.add_argument(
    "--lidar_mesh_0",
    type=str,
    default="/World/envs/env_0/blue_bin_0",
    help="First mesh search root for RayCaster lidar_0 (searches descendants for Mesh prims).",
)
parser.add_argument(
    "--lidar_mesh_1",
    type=str,
    default="/World/envs/env_0/blue_bin_0",
    help="Second mesh search root for RayCaster lidar_1 (searches descendants for Mesh prims).",
)
parser.add_argument(
    "--lidar_mesh_paths",
    type=str,
    nargs="+",
    default=None,
    help="Optional list of mesh prim paths. If set, overrides --lidar_mesh_0/1 and supports any number of obstacles.",
)
parser.add_argument(
    "--lidar_objects",
    type=str,
    nargs="+",
    default=None,
    help="Object names like chair_0 table_0; auto-converted to <env_prim_root>/<name>.",
)
parser.add_argument(
    "--env_prim_root",
    type=str,
    default="/World/envs/env_0",
    help="Root prim path used when --lidar_objects are provided.",
)
parser.add_argument(
    "--visual_mode",
    type=str,
    default="depth_rgb",
    choices=["off", "depth_rgb", "lidar_rgb", "rtx_rgb"],
    help=(
        "Image source. Default depth_rgb (cluster-safe, no NGX). "
        "rtx_rgb loads lab 3DGS but needs RTX (--enable_cameras); often fails on compute2."
    ),
)
parser.add_argument(
    "--img_height",
    type=int,
    default=640,
    help="Sensor height in pixels (portrait 640×480 before CCW rotation → landscape PNG).",
)
parser.add_argument(
    "--img_width",
    type=int,
    default=480,
    help="Sensor width in pixels.",
)
args_cli = parser.parse_args()

from visual_obs_utils import configure_app_for_visual, resolve_visual_mode

_visual_mode = resolve_visual_mode(args_cli)
configure_app_for_visual(args_cli, _visual_mode)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------
# Imports after launching app
# ---------------------------------------------------------------------
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg_PLAY
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, RayCasterCameraCfg, patterns
from visual_obs_utils import (
    build_depth_camera_cfgs,
    build_rtx_camera_cfg,
    depth_to_rgb,
    lidar_ranges_to_rgb,
    merge_depth_maps_multi,
)

import isaaclab.sim as sim_utils
from pxr import UsdGeom, Gf, Sdf
import omni.usd

from data_collection_obstacles import add_blue_bin, add_table , add_chair
from lab_scene_utils import (
    default_obstacle_mesh_paths,
    default_raycast_mesh_paths,
    load_lab_scene_usd,
    rotate_sensor_ccw_to_landscape,
)
from camera_pose_utils import camera_world_pose

ISAACLAB_LEG_IDXS = torch.tensor([
    0, 3, 7, 11, 15, 19,
    1, 4, 8, 12, 16, 20
])


def _object_to_mesh_path(object_name: str, env_prim_root: str) -> str:
    """Convert obstacle object name to mesh prim path."""
    obj_name = object_name.strip()
    if not obj_name:
        raise ValueError("Empty object name in --lidar_objects is not allowed.")
    if obj_name.startswith("/"):
        return obj_name
    return f"{env_prim_root.rstrip('/')}/{obj_name}"


def _merge_ray_hits_multi(
    origin_np: np.ndarray,
    hits_list: list[np.ndarray],
    max_d: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge k (N,3) world hit arrays from sensors with same origin. Returns (diff_w, ranges, ranges_xy)."""
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


class G1TurningCollectorDual:
    """Collect locomotion data with multi RayCaster sensors merged per ray."""

    @staticmethod
    def _waypoints_to_list(waypoint) -> list:
        w = np.asarray(waypoint, dtype=np.float64)
        if w.ndim == 1 and w.size == 2:
            return [w]
        if w.ndim == 2 and w.shape[1] == 2:
            return [w[i] for i in range(w.shape[0])]
        raise ValueError(f"waypoint must be shape (2,) or (N, 2), got {w.shape}")

    def __init__(self, vx=0.5, vy=0.0, yaw_rate=0.0,
                 waypoint=(2.0, 1.0), img_res=(640, 480),
                 save_every=10, collect_data=True,
                 lidar_mesh_0: str | None = None,
                 lidar_mesh_1: str | None = None,
                 obstacle_names: tuple[str, ...] = ("blue_bin_0",)):
        TASK = "Isaac-Velocity-Flat-G1-v0"
        RL_LIBRARY = "rsl_rl"
        self.collect_data = collect_data
        self.waypoint = np.array(waypoint)
        self.visual_mode = getattr(args_cli, "visual_mode", "depth_rgb")
        self.collect_visual = self.visual_mode != "off"
        self.img_res = img_res

        if args_cli.lidar_objects is not None and len(args_cli.lidar_objects) > 0:
            mesh_paths = tuple(
                _object_to_mesh_path(obj_name, args_cli.env_prim_root)
                for obj_name in args_cli.lidar_objects
            )
        elif args_cli.lidar_mesh_paths is not None and len(args_cli.lidar_mesh_paths) > 0:
            mesh_paths = tuple(args_cli.lidar_mesh_paths)
        elif lidar_mesh_0 is not None or lidar_mesh_1 is not None:
            mesh0 = lidar_mesh_0 if lidar_mesh_0 is not None else args_cli.lidar_mesh_0
            mesh1 = lidar_mesh_1 if lidar_mesh_1 is not None else args_cli.lidar_mesh_1
            mesh_paths = (mesh0, mesh1)
        elif self.visual_mode == "depth_rgb":
            load_lab_scene_usd()
            mesh_paths = default_raycast_mesh_paths(
                args_cli.env_prim_root,
                obstacle_names=obstacle_names,
                include_lab_scene=True,
            )
        else:
            mesh_paths = default_obstacle_mesh_paths(
                args_cli.env_prim_root,
                obstacle_names=obstacle_names,
            )
        self._lidar_mesh_paths = mesh_paths

        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
        checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)

        env_cfg = G1FlatEnvCfg_PLAY()
        env_cfg.scene.num_envs = 1
        env_cfg.episode_length_s = 100000
        env_cfg.curriculum = None
        env_cfg.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
        env_cfg.decimation = 1
        env_cfg.sim.render_interval = 1

        env_cfg.terminations.base_contact = None ###WE ADDED THIS HERE TO FIX ENV RESET WHEN WE HIT TORSO OF HUMANOID AND OBSTALE

        # --- Add obstacles below (see data_collection_obstacles.py) ---
        # add_obstacle_cube(env_cfg, pos=(2, 0.0, 5.25), size=(0.5, 1.0, 0.5), index=0)
        add_blue_bin(env_cfg, pos=(2, 0, 0.5), index=0)
        #add_table(env_cfg, pos=(4, 0, 0.5), index=0)
        #add_chair(env_cfg, pos=(2, 0, 0.5), index=0)


        # --- Add obstacles above ---

        env_cfg.scene.robot_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*link.*",
            update_period=0.0,
            debug_vis=True,
            filter_prim_paths_expr=[],
        )

        self.camera_fps = 15.0
        self.lidar_fps = 7.0
        self.camera_period_s = 1.0 / self.camera_fps
        self.lidar_period_s = 1.0 / self.lidar_fps

        self._depth_cam_names: list[str] = []
        if self.visual_mode == "rtx_rgb":
            env_cfg.scene.camera = build_rtx_camera_cfg(
                img_res=img_res,
                update_period_s=self.camera_period_s,
                sim_utils=sim_utils,
                camera_cfg_cls=CameraCfg,
            )
            print("[INFO] rtx_rgb: CameraCfg + PathTracing (3DGS lab, --enable_cameras).")
        elif self.visual_mode == "depth_rgb":
            for name, dc_cfg in build_depth_camera_cfgs(
                self._lidar_mesh_paths,
                img_res=img_res,
                update_period_s=self.camera_period_s,
                patterns_mod=patterns,
                ray_caster_camera_cfg_cls=RayCasterCameraCfg,
            ):
                setattr(env_cfg.scene, name, dc_cfg)
                self._depth_cam_names.append(name)
            print(
                f"[INFO] depth_rgb: {len(self._depth_cam_names)} RayCasterCamera sensors "
                f"(no --enable_cameras / no NGX)."
            )
        elif self.visual_mode == "lidar_rgb":
            print("[INFO] lidar_rgb: range-image pseudo-RGB from merged LiDAR (no RTX).")
        else:
            print("[INFO] visual_mode=off: no images saved.")

        _lidar_common = dict(
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

        # Isaac RayCaster only allows one mesh per sensor; create one lidar per mesh and merge in Python.
        self._lidar_sensor_names = []
        for i, mesh_path in enumerate(self._lidar_mesh_paths):
            sensor_name = f"lidar_{i}"
            setattr(env_cfg.scene, sensor_name, RayCasterCfg(mesh_prim_paths=[mesh_path], **_lidar_common))
            self._lidar_sensor_names.append(sensor_name)
            print(f"[INFO] LiDAR mesh path: {sensor_name} -> {mesh_path}")

        self.env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
        self.device = self.env.unwrapped.device
        self.sim_dt = float(self.env.unwrapped.cfg.sim.dt)
        self.sim_hz = 1.0 / self.sim_dt
        print(f"[INFO] Control/data loop set to {self.sim_hz:.1f} Hz (dt={self.sim_dt:.4f}s)")
        if self.collect_visual:
            print(f"[INFO] Visual ({self.visual_mode}) / LiDAR rates: {self.camera_fps:.1f} Hz / {self.lidar_fps:.1f} Hz")
        else:
            print(f"[INFO] LiDAR target rate: {self.lidar_fps:.1f} Hz")
        self.next_camera_time_s = 0.0
        self.next_lidar_time_s = 0.0
        self.camera_frame_idx = 0
        self.lidar_frame_idx = 0

        # RTX path: load lab after env (same as DataCollection_loop). depth_rgb loads earlier.
        if self.visual_mode != "depth_rgb":
            load_lab_scene_usd()

        runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=self.device)
        runner.load(checkpoint)
        self.policy = runner.get_inference_policy(device=self.device)

        self.commands = torch.zeros(1, 3, device=self.device)
        self.commands[:, 0] = vx
        self.commands[:, 1] = vy
        self.commands[:, 2] = yaw_rate

        self.max_speed = float(np.linalg.norm([vx, vy]))
        self.save_every = save_every

        data_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.base_dir = os.path.join(data_parent, run_stamp)
        os.makedirs(self.base_dir, exist_ok=True)
        if self.collect_visual:
            self.image_dir = os.path.join(self.base_dir, "images")
            os.makedirs(self.image_dir, exist_ok=True)
        self.lidar_dir = os.path.join(self.base_dir, "lidar")
        os.makedirs(self.lidar_dir, exist_ok=True)
        self.save_path = os.path.join(self.base_dir, "locomotion_dataset.csv")
        print(f"[INFO] Trajectory folder: {self.base_dir}")

        robot = self.env.unwrapped.scene["robot"]
        self.num_joints = robot.data.joint_pos.shape[1]
        print(f"[INFO] Detected {self.num_joints} actuated joints. Waypoint = {self.waypoint}")

        if self.visual_mode == "rtx_rgb":
            self.camera = self.env.unwrapped.scene["camera"]
        self._head_body_idx: int | None = None
        head_names = ("head_link", "Head", "head")
        body_names = list(self.env.unwrapped.scene["robot"].data.body_names)
        for hn in head_names:
            if hn in body_names:
                self._head_body_idx = body_names.index(hn)
                break
        if self.collect_visual and self._head_body_idx is None:
            print("[WARN] head_link not found in body_names — camera_poses.csv will use root pose.")
        print(f"[INFO] visual_mode={self.visual_mode}, collect_data={self.collect_data}")
        self._ignored_collision_links = {"left_ankle_roll_link", "right_ankle_roll_link"}
        self._collision_force_threshold = 0.1

        self._add_waypoint_marker()

    def _load_scene_usd(self):
        """Backward-compatible alias; scene is loaded before env init."""
        load_lab_scene_usd(verbose=False)

    def _add_waypoint_marker(self):
        stage = self.env.unwrapped.scene.stage

        waypoints = self._waypoints_to_list(self.waypoint)

        for i, wp in enumerate(waypoints):
            sphere_path = Sdf.Path(f"/World/WaypointMarker_{i}")
            if stage.GetPrimAtPath(sphere_path):
                continue
            sphere = UsdGeom.Sphere.Define(stage, sphere_path)
            sphere.GetRadiusAttr().Set(0.1)
            sphere.AddTranslateOp().Set(Gf.Vec3f(wp[0], wp[1], 0.0))
            sphere.CreateVisibilityAttr().Set("invisible")
        print(f"[INFO] Added {len(waypoints)} waypoint markers.")

    def quat_to_yaw(self, quat):
        w, x, y, z = quat
        yaw = np.arctan2(2.0 * (w * z + x * y),
                        1.0 - 2.0 * (y * y + z * z))
        return yaw

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def run(self, num_steps=3000):
        obs, _ = self.env.reset()

        scene = self.env.unwrapped.scene
        robot = scene["robot"]

        root_pose = torch.tensor([[0.0, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0]], device=self.device)
        robot.write_root_pose_to_sim(root_pose)

        waypoints = self._waypoints_to_list(self.waypoint)

        print(f"[INFO] Running {num_steps} steps through {len(waypoints)} waypoints: {waypoints}")

        stop_thresh = 0.05
        k_yaw = 1.0
        max_yaw_rate = 1.0
        yaw_smooth = 0.1
        prev_yaw_rate = 0.0
        current_target_idx = 0
        threshold_deg = 55
        target = np.array(waypoints[current_target_idx])

        prev_yaw = 0.0
        prev_theta_v = 0.0

        lidars = [scene[name] for name in self._lidar_sensor_names]

        if self.collect_data:
            f = open(self.save_path, mode="w", newline="")
            writer = csv.writer(f)
            N = self.num_joints
            header = (
                ["sim_step", "sim_time_s"] +
                ["px", "py", "pz"] +
                [f"base_quat_{i}" for i in range(4)] +
                ["base_lin_vel_x", "base_lin_vel_y", "base_lin_vel_z"] +
                ["base_ang_vel_x", "base_ang_vel_y", "base_ang_vel_z"] +
                [f"joint_pos_{i}" for i in range(N)] +
                [f"joint_vel_{i}" for i in range(N)] +
                [f"torque_{i}" for i in range(N)] +
                [f"action_{i}" for i in range(N)] +
                ["vx_cmd", "vy_cmd", "yaw_rate_cmd"] +
                ["target_x", "target_y"] +
                ["lidar_min_distance"] +
                ["lidar_min_distance_xy"] +
                ["is_collision"]
            )
            writer.writerow(header)

            sensor = self.env.unwrapped.scene["robot_contact"]
            link_names = sensor.body_names
            self._collision_link_indices = [
                i for i, name in enumerate(link_names)
                if name not in self._ignored_collision_links
            ]

            self.contact_file = open(
                os.path.join(self.base_dir, "contact_force.csv"),
                "w", newline=""
            )
            self.contact_writer = csv.writer(self.contact_file)

            contact_header = ["step"]
            for name in link_names:
                contact_header += [f"{name}_x", f"{name}_y", f"{name}_z"]

            self.contact_writer.writerow(contact_header)
            self.contact_file.flush()
            print(f"[INFO] contact_force.csv opened ({len(link_names)} links): {self.contact_file.name}")

            if self.collect_visual:
                self.camera_poses_path = os.path.join(self.base_dir, "camera_poses.csv")
                self.camera_poses_file = open(self.camera_poses_path, "w", newline="")
                self.camera_poses_writer = csv.writer(self.camera_poses_file)
                self.camera_poses_writer.writerow(
                    [
                        "frame_idx",
                        "sim_time_s",
                        "cam_px",
                        "cam_py",
                        "cam_pz",
                        "cam_qw",
                        "cam_qx",
                        "cam_qy",
                        "cam_qz",
                    ]
                )
                print(f"[INFO] camera_poses.csv for offline GS RGB: {self.camera_poses_path}")

        else:
            writer = None
            self.camera_poses_writer = None

        for step in range(num_steps):
            robot = self.env.unwrapped.scene["robot"]
            data = robot.data

            base_pos = data.root_pos_w[0].cpu().numpy()
            base_quat = data.root_quat_w[0].cpu().numpy()

            yaw = self.quat_to_yaw(base_quat)
            yaw = np.unwrap([prev_yaw, yaw])[1]
            prev_yaw = yaw

            dx = target[0] - base_pos[0]
            dy = target[1] - base_pos[1]
            dist = np.sqrt(dx**2 + dy**2)

            if dist < stop_thresh:
                current_target_idx += 1
                if current_target_idx >= len(waypoints):
                    break
                else:
                    target = np.array(waypoints[current_target_idx])
                    self.waypoint = target
                    self._add_waypoint_marker()
                    continue

            local_dx = np.cos(yaw) * dx + np.sin(yaw) * dy
            local_dy = -np.sin(yaw) * dx + np.cos(yaw) * dy

            direction_local = np.array([local_dx, local_dy])
            direction_local /= np.linalg.norm(direction_local)

            vx_local = self.max_speed * direction_local[0]
            vy_local = self.max_speed * direction_local[1]

            theta_v = np.arctan2(vy_local, vx_local)
            theta_v = (theta_v + np.pi) % (2 * np.pi) - np.pi

            theta_deg = np.degrees(theta_v)

            dead_zone_deg = 30
            dead_zone_start = 180 - dead_zone_deg

            if -threshold_deg <= theta_deg <= threshold_deg:
                yaw_rate_to_use = k_yaw * theta_v
                vx_cmd = vx_local
                vy_cmd = vy_local

            else:
                vx_cmd = 0.1
                vy_cmd = 0.0
                yaw_smooth = 1.0

                if abs(theta_deg) >= dead_zone_start:
                    yaw_rate_to_use = +max_yaw_rate
                else:
                    yaw_rate_to_use = k_yaw * theta_v
                    yaw_rate_to_use = np.clip(yaw_rate_to_use, -max_yaw_rate, max_yaw_rate)

            yaw_rate = (1 - yaw_smooth) * prev_yaw_rate + yaw_smooth * yaw_rate_to_use
            prev_yaw_rate = yaw_rate

            alpha = 1
            prev_vx = self.commands[0, 0].item()
            prev_vy = self.commands[0, 1].item()

            vx_cmd = (1 - alpha) * prev_vx + alpha * vx_cmd
            vy_cmd = (1 - alpha) * prev_vy + alpha * vy_cmd

            target_cmd = torch.tensor([[vx_cmd, vy_cmd, yaw_rate]], device=self.device)
            if step == 0:
                self.commands = target_cmd.clone()

            self.commands = target_cmd.clone()

            self.env.unwrapped.command_manager._terms["base_velocity"].command[:] = self.commands.clone()
            obs["policy"][0, 11] = self.commands[0, 2]

            print(f"obs: {obs}")
            vec = obs["policy"][0]

            print("base_lin_vel:", vec[0:3])
            print("base_ang_vel:", vec[3:6])
            print("proj_gravity:", vec[6:9])
            print("commands:", vec[9:12])
            print("joint_pos:", vec[12:49])
            print("joint_vel:", vec[49:86])
            print("actions:", vec[86:123])

            with torch.inference_mode():
                actions = self.policy(obs)
                print(f"actions: {actions}")

            obs, _, _, _ = self.env.step(actions)

            sensor = self.env.unwrapped.scene["robot_contact"]

            print("Contact Force:")
            print(sensor.body_names)
            print(sensor.data.net_forces_w)

            lidar_points_list = [lidar.data.ray_hits_w[0].detach().cpu().numpy() for lidar in lidars]
            lidar_origin_w = lidars[0].data.pos_w[0].detach().cpu().numpy()

            max_d = float(getattr(lidars[0].cfg, "max_distance", 1e6))
            diff_w, ranges, ranges_xy = _merge_ray_hits_multi(
                lidar_origin_w, lidar_points_list, max_d
            )

            finite_hit = np.isfinite(diff_w).all(axis=1) & (ranges > 1e-4) & (ranges < max_d * 0.999)
            valid_ray = finite_hit

            print(f"lidar merged (first 3 diff): {diff_w[:3]}")
            print(f"lidar origin_w: {lidar_origin_w}, base_pos: {base_pos}")
            print(f"ranges (first 3): {ranges[:3]}")

            positive_ranges = ranges[valid_ray]
            lidar_min_range_nonzero_m = (
                float(np.min(positive_ranges))
                if positive_ranges.size > 0
                else float("nan")
            )

            positive_ranges_xy = ranges_xy[valid_ray]
            lidar_min_range_xy_nonzero_m = (
                float(np.min(positive_ranges_xy))
                if positive_ranges_xy.size > 0
                else float("nan")
            )

            print("lidar shape:", diff_w.shape)
            print("lidar min/max:", ranges.min(), ranges.max())

            print(f"[STEP {step}] Target={target}, dist={dist:.2f}, yaw={np.degrees(yaw):.1f}°, "
                  f"vx={vx_cmd:.2f}, vy={vy_cmd:.2f}, yaw_rate={np.degrees(yaw_rate):.1f}°/s")

            if self.collect_data:
                assert writer is not None
                base_lin_vel = data.root_lin_vel_w[0].cpu().numpy()
                base_ang_vel = data.root_ang_vel_w[0].cpu().numpy()
                joint_pos = data.joint_pos[0, :self.num_joints].cpu().numpy()
                joint_vel = data.joint_vel[0, :self.num_joints].cpu().numpy()
                torques = data.applied_torque[0, :self.num_joints].cpu().numpy()
                actions_np = actions[0, :self.num_joints].detach().cpu().numpy()
                commands_np = self.commands[0].detach().cpu().numpy()
                sim_step = float(step)
                sim_time_s = step * self.sim_dt
                contact = sensor.data.net_forces_w[0].detach().cpu().numpy()

                row = np.concatenate([
                    np.array([sim_step, sim_time_s], dtype=np.float64),
                    base_pos, base_quat, base_lin_vel, base_ang_vel,
                    joint_pos, joint_vel, torques, actions_np, commands_np, target,
                    np.array(
                        [
                            lidar_min_range_nonzero_m,
                            lidar_min_range_xy_nonzero_m,
                            float(
                                np.any(
                                    np.abs(contact[self._collision_link_indices, :])
                                    > self._collision_force_threshold
                                )
                            ),
                        ],
                        dtype=np.float64,
                    ),
                ])
                writer.writerow(row.tolist())

                contact_row = [step]
                for i in range(contact.shape[0]):
                    contact_row += contact[i].tolist()

                self.contact_writer.writerow(contact_row)
                if step % 50 == 0:
                    self.contact_file.flush()

                if self.collect_visual and sim_time_s + 1e-12 >= self.next_camera_time_s:
                    import imageio

                    if self.visual_mode == "rtx_rgb":
                        rgb_tensor = self.camera.data.output["rgb"][0]
                        rgb_np = rgb_tensor[..., :3].cpu().numpy()
                        if rgb_np.dtype != np.uint8:
                            rgb_np = (rgb_np * 255).clip(0, 255).astype(np.uint8)
                        rgb_np = rotate_sensor_ccw_to_landscape(rgb_np)
                    elif self.visual_mode == "depth_rgb":
                        scene = self.env.unwrapped.scene
                        depth_list = [
                            scene[n].data.output["distance_to_image_plane"][0].detach().cpu().numpy()
                            for n in self._depth_cam_names
                        ]
                        merged = merge_depth_maps_multi(depth_list, max_d=10.0)
                        rgb_np = rotate_sensor_ccw_to_landscape(depth_to_rgb(merged))
                    elif self.visual_mode == "lidar_rgb":
                        rgb_np = rotate_sensor_ccw_to_landscape(
                            lidar_ranges_to_rgb(
                                ranges, valid_ray, out_size=self.img_res
                            )
                        )
                    else:
                        rgb_np = None

                    if rgb_np is not None:
                        imageio.imwrite(
                            os.path.join(self.image_dir, f"rgb_{self.camera_frame_idx:06d}.png"),
                            rgb_np,
                        )
                        if getattr(self, "camera_poses_writer", None) is not None:
                            if self._head_body_idx is not None:
                                head_pos = data.body_pos_w[0, self._head_body_idx].cpu().numpy()
                                head_quat = data.body_quat_w[0, self._head_body_idx].cpu().numpy()
                            else:
                                head_pos = base_pos
                                head_quat = base_quat
                            cam_pos, cam_quat = camera_world_pose(head_pos, head_quat)
                            self.camera_poses_writer.writerow(
                                [
                                    self.camera_frame_idx,
                                    sim_time_s,
                                    cam_pos[0],
                                    cam_pos[1],
                                    cam_pos[2],
                                    cam_quat[0],
                                    cam_quat[1],
                                    cam_quat[2],
                                    cam_quat[3],
                                ]
                            )
                        self.camera_frame_idx += 1
                        self.next_camera_time_s += self.camera_period_s

                if sim_time_s + 1e-12 >= self.next_lidar_time_s:
                    np.save(
                        os.path.join(self.lidar_dir, f"lidar_{self.lidar_frame_idx:06d}.npy"),
                        diff_w,
                    )
                    self.lidar_frame_idx += 1
                    self.next_lidar_time_s += self.lidar_period_s

        if self.collect_data and writer is not None:
            f.flush()
            f.close()
            if getattr(self, "contact_file", None) is not None:
                self.contact_file.flush()
                self.contact_file.close()
                print(f"[INFO] contact_force.csv saved")
            if getattr(self, "camera_poses_file", None) is not None:
                self.camera_poses_file.flush()
                self.camera_poses_file.close()
                print(f"[INFO] camera_poses.csv saved (use run_gs_rgb_offline.sh for true GS RGB on H100)")
            print(f"[INFO] Trajectory folder: {os.path.abspath(self.base_dir)}")
            print(f"[INFO] Dataset saved to: {os.path.abspath(self.save_path)}")
            if self.collect_visual:
                print(f"[INFO] Images written to: {os.path.abspath(self.image_dir)}")
            print(f"[INFO] LiDAR written to: {os.path.abspath(self.lidar_dir)}")


def main():
    collect_flag = not args_cli.no_collect
    img_res = (args_cli.img_height, args_cli.img_width)

    collector = G1TurningCollectorDual(
        vx=args_cli.vx,
        vy=args_cli.vy,
        yaw_rate=args_cli.yaw_rate,
        waypoint=[(0, 0), (2, 0.7), (3, 0)],
        img_res=img_res,
        save_every=1,
        collect_data=collect_flag,
    )

    collector.run(num_steps=60000)
    simulation_app.close()


if __name__ == "__main__":
    main()
