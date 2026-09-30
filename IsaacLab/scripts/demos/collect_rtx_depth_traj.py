"""Record a full avoid trajectory: FPV RGB|depth + third-person RGB.

Same stack as collect_rtx_depth.py (G1, GS lab, blue bin, RTX head camera).
The robot follows a fixed waypoint sequence around the bin. A world-fixed
third-person camera is placed from the path + bin geometry (side-elevated,
opposite the detour) so the whole avoid is in frame.

Usage (from IsaacLab/):

    .\\isaaclab.bat -p scripts/demos/collect_rtx_depth_traj.py --no_headless

    .\\isaaclab.bat -p scripts/demos/collect_rtx_depth_traj.py --headless

    # Custom path
    .\\isaaclab.bat -p scripts/demos/collect_rtx_depth_traj.py --no_headless \\
        --bin_xy 4,0 --waypoints 0,0;2,0;4,-0.75;5,0
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="Follow a waypoint avoid path; dump FPV rgb|depth and third-person RGB."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
parser.add_argument(
    "--no_headless",
    action="store_false",
    dest="headless",
    help="Open the Isaac GUI. Default is headless.",
)
parser.add_argument(
    "--duration_s",
    type=float,
    default=40.0,
    help="Max sim time (s). Stops earlier when the last waypoint is reached.",
)
parser.add_argument(
    "--video_fps",
    type=float,
    default=15.0,
    help="How many frames to grab per second of sim time.",
)
parser.add_argument(
    "--playback_fps",
    type=float,
    default=8.0,
    help="Output video fps. Lower than --video_fps plays slower than real time.",
)
parser.add_argument("--img_height", type=int, default=640, help="Portrait FPV sensor height.")
parser.add_argument("--img_width", type=int, default=480, help="Portrait FPV sensor width.")
parser.add_argument("--tp_height_px", type=int, default=480, help="Third-person image height.")
parser.add_argument("--tp_width_px", type=int, default=854, help="Third-person image width.")
parser.add_argument("--max_speed", type=float, default=0.45)
parser.add_argument("--max_depth", type=float, default=10.0)
parser.add_argument(
    "--bin_xy",
    type=str,
    default="4,0",
    help="Blue bin XY as 'x,y'.",
)
parser.add_argument(
    "--waypoints",
    type=str,
    default="0,0;2,0;4,-0.75;5,0",
    help="Waypoint XY list 'x,y;x,y;...'. First point is spawn.",
)
parser.add_argument("--wp_stop_m", type=float, default=0.22, help="Reach radius (m).")
parser.add_argument(
    "--tp_height",
    type=float,
    default=2.55,
    help="Third-person camera height (m).",
)
parser.add_argument(
    "--tp_side_m",
    type=float,
    default=0.0,
    help="Distance to the side of the path (m). 0 = auto from path length.",
)
parser.add_argument(
    "--tp_side_sign",
    type=float,
    default=-1.0,
    help="Camera side vs start→goal: -1 = right/open floor (default), +1 = left/desks.",
)
parser.add_argument(
    "--tp_look_z",
    type=float,
    default=0.55,
    help="Look-at height (m).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="Output folder. Default: IsaacLab/data/rtx_depth_traj/<timestamp>/",
)
parser.add_argument("--save_frames", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--nurec_deyellow", type=float, default=0.55)
parser.add_argument("--no_nurec_ccm", action="store_true")
parser.add_argument("--no_depth_proxies", action="store_true")
args_cli = parser.parse_args()
args_cli.visual_mode = "rtx_rgb"

from visual_obs_utils import configure_app_for_visual, resolve_visual_mode

_visual_mode = resolve_visual_mode(args_cli)
configure_app_for_visual(args_cli, _visual_mode)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import imageio
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors.camera import CameraCfg
from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg_PLAY,
)

from data_collection_obstacles import add_bin_fill_light, add_blue_bin, apply_blue_bin_visual
from lab_scene_utils import (
    apply_nurec_floor_color,
    load_lab_scene_usd,
    rotate_sensor_ccw_to_landscape,
    set_lab_depth_proxies_visible,
)
from visual_obs_utils import build_rtx_camera_cfg, depth_to_gray_rgb

TASK = "Isaac-Velocity-Flat-G1-v0"
RL_LIBRARY = "rsl_rl"


def _parse_xy(s: str) -> np.ndarray:
    parts = [p.strip() for p in str(s).replace("/", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected 'x,y', got {s!r}")
    return np.array([float(parts[0]), float(parts[1])], dtype=np.float64)


def _parse_waypoints(s: str) -> np.ndarray:
    chunks = [c.strip() for c in str(s).replace("/", ";").split(";") if c.strip()]
    pts = np.stack([_parse_xy(c) for c in chunks], axis=0)
    if pts.shape[0] < 2:
        raise ValueError("Need at least two waypoints (spawn and goal).")
    return pts


def quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def yaw_to_quat_wxyz(yaw: float) -> tuple[float, float, float, float]:
    h = 0.5 * float(yaw)
    return (float(np.cos(h)), 0.0, 0.0, float(np.sin(h)))


class ShortestYawWander:
    """Same shortest-yaw tracker as DataCollection / collect_rtx_depth."""

    def __init__(self, max_speed: float = 0.45, stop_thresh: float = 0.22):
        self.max_speed = float(max_speed)
        self.stop_thresh = float(stop_thresh)
        self.k_yaw = 1.0
        self.max_yaw_rate = 1.0
        self.threshold_deg = 55.0
        self.dead_zone_deg = 30.0
        self.prev_yaw_rate = 0.0
        self.prev_yaw = 0.0

    def command(self, base_pos: np.ndarray, base_quat: np.ndarray, target_xy: np.ndarray):
        yaw = quat_to_yaw(base_quat)
        yaw = float(np.unwrap([self.prev_yaw, yaw])[1])
        self.prev_yaw = yaw

        dx = float(target_xy[0] - base_pos[0])
        dy = float(target_xy[1] - base_pos[1])
        dist = float(np.hypot(dx, dy))
        if dist < self.stop_thresh:
            return np.zeros(3, dtype=np.float32), dist, True

        local_dx = np.cos(yaw) * dx + np.sin(yaw) * dy
        local_dy = -np.sin(yaw) * dx + np.cos(yaw) * dy
        direction = np.array([local_dx, local_dy], dtype=np.float64)
        direction /= np.linalg.norm(direction) + 1e-8

        vx_local = self.max_speed * direction[0]
        vy_local = self.max_speed * direction[1]
        theta_v = np.arctan2(vy_local, vx_local)
        theta_v = (theta_v + np.pi) % (2.0 * np.pi) - np.pi
        theta_deg = float(np.degrees(theta_v))
        dead_zone_start = 180.0 - self.dead_zone_deg

        if -self.threshold_deg <= theta_deg <= self.threshold_deg:
            yaw_rate_to_use = self.k_yaw * theta_v
            vx_cmd, vy_cmd = vx_local, vy_local
            yaw_smooth = 0.1
        else:
            vx_cmd, vy_cmd = 0.1, 0.0
            yaw_smooth = 1.0
            if abs(theta_deg) >= dead_zone_start:
                yaw_rate_to_use = self.max_yaw_rate
            else:
                yaw_rate_to_use = float(
                    np.clip(self.k_yaw * theta_v, -self.max_yaw_rate, self.max_yaw_rate)
                )

        yaw_rate = (1.0 - yaw_smooth) * self.prev_yaw_rate + yaw_smooth * yaw_rate_to_use
        self.prev_yaw_rate = float(yaw_rate)
        return np.array([vx_cmd, vy_cmd, yaw_rate], dtype=np.float32), dist, False


def third_person_eye_target(
    waypoints: np.ndarray,
    bin_xy: np.ndarray,
    *,
    height: float,
    side_m: float,
    look_z: float,
    side_sign: float = -1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Elevated side camera over the open floor, whole path + bin in frame.

    Look-at and distance still come from the waypoint/bin bbox. The side is
    locked (default: right of start→goal, -Y on this aisle) so a -Y detour
    does not slam the camera into the desk wall.
    """
    pts = np.vstack([waypoints, bin_xy.reshape(1, 2)])
    look_xy = pts.mean(axis=0)
    tangent = waypoints[-1] - waypoints[0]
    tnorm = float(np.linalg.norm(tangent))
    if tnorm < 1e-6:
        tangent = np.array([1.0, 0.0], dtype=np.float64)
    else:
        tangent = tangent / tnorm
    perp = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    sign = -1.0 if float(side_sign) < 0.0 else 1.0
    cam_side = sign * perp
    span = max(tnorm, 1.0)
    dist = float(side_m) if side_m > 1e-6 else max(4.8, 0.95 * span + 1.8)
    eye_xy = look_xy + cam_side * dist - tangent * (0.22 * span)
    eye = np.array([eye_xy[0], eye_xy[1], float(height)], dtype=np.float64)
    target = np.array([look_xy[0], look_xy[1], float(look_z)], dtype=np.float64)
    return eye, target


def lookat_opengl_quat(eye: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    eyes = torch.tensor(np.asarray(eye, dtype=np.float32).reshape(1, 3))
    targets = torch.tensor(np.asarray(target, dtype=np.float32).reshape(1, 3))
    rot = create_rotation_matrix_from_view(eyes, targets, up_axis="Z")
    quat = quat_from_matrix(rot)[0]
    return tuple(float(v) for v in quat.tolist())


def build_tp_camera_cfg(
    *,
    eye: np.ndarray,
    target: np.ndarray,
    height: int,
    width: int,
    sim_utils_mod,
    camera_cfg_cls,
):
    quat = lookat_opengl_quat(eye, target)
    return camera_cfg_cls(
        prim_path="{ENV_REGEX_NS}/tp_camera",
        update_period=0.0,
        height=int(height),
        width=int(width),
        data_types=["rgb"],
        spawn=sim_utils_mod.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=24.0,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=camera_cfg_cls.OffsetCfg(
            pos=(float(eye[0]), float(eye[1]), float(eye[2])),
            rot=quat,
            convention="opengl",
        ),
    )


def _rgb_to_uint8(rgb_np: np.ndarray) -> np.ndarray:
    if rgb_np.dtype != np.uint8:
        rgb_np = (np.clip(rgb_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgb_np


def _squeeze_hw(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    while out.ndim > 2 and out.shape[-1] == 1:
        out = out[..., 0]
    if out.ndim != 2:
        raise ValueError(f"Expected HxW depth, got {out.shape}")
    return out


def _refresh_camera(camera, sim, n_renders: int = 2) -> None:
    for _ in range(n_renders):
        sim.render()
    if hasattr(camera, "_is_outdated"):
        camera._is_outdated[:] = True
    camera.update(0.0, force_recompute=True)


def _rgb_from_camera(camera, *, rotate: bool) -> np.ndarray:
    rgb_tensor = camera.data.output["rgb"][0]
    rgb_np = _rgb_to_uint8(rgb_tensor[..., :3].detach().cpu().numpy())
    if rotate:
        return rotate_sensor_ccw_to_landscape(rgb_np)
    return np.ascontiguousarray(rgb_np)


def _depth_from_camera(camera, max_depth: float) -> np.ndarray:
    depth_tensor = camera.data.output["distance_to_image_plane"][0]
    depth_m = _squeeze_hw(depth_tensor.detach().cpu().numpy()).astype(np.float32)
    depth_m = rotate_sensor_ccw_to_landscape(depth_m)
    return depth_to_gray_rgb(depth_m, max_depth=max_depth)


def _read_with_retry(read_fn, env, commands, zero_actions, label: str):
    last_exc: Exception | None = None
    for attempt in range(15):
        try:
            return read_fn()
        except TypeError as exc:
            last_exc = exc
            env.unwrapped.command_manager._terms["base_velocity"].command[:] = commands
            env.step(zero_actions)
            if attempt == 0 or attempt == 14:
                print(f"[WARN] {label} read failed (attempt {attempt + 1}/15): {exc}")
    raise RuntimeError(f"{label} never produced a frame.") from last_exc


def _hstack_rgb(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[0] != right.shape[0]:
        raise ValueError(f"Height mismatch: {left.shape} vs {right.shape}")
    return np.concatenate([left, right], axis=1)


def _open_video_writer(path: str, fps: float):
    kwargs_list = (
        {"fps": fps, "codec": "libx264", "quality": 8},
        {"fps": fps, "macro_block_size": 1},
        {"fps": fps},
    )
    last_exc: Exception | None = None
    for kwargs in kwargs_list:
        try:
            return imageio.get_writer(path, **kwargs)
        except (TypeError, ValueError, RuntimeError) as exc:
            last_exc = exc
    raise RuntimeError(f"Could not open video writer for {path}") from last_exc


def main() -> None:
    waypoints = _parse_waypoints(args_cli.waypoints)
    bin_xy = _parse_xy(args_cli.bin_xy)
    img_res = (int(args_cli.img_height), int(args_cli.img_width))
    max_depth = float(args_cli.max_depth)
    use_depth_proxies = not bool(args_cli.no_depth_proxies)

    tp_eye, tp_target = third_person_eye_target(
        waypoints,
        bin_xy,
        height=float(args_cli.tp_height),
        side_m=float(args_cli.tp_side_m),
        look_z=float(args_cli.tp_look_z),
        side_sign=float(args_cli.tp_side_sign),
    )

    if args_cli.output_dir:
        out_dir = os.path.abspath(args_cli.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/rtx_depth_traj", stamp)
        )
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    if args_cli.save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    side_path = os.path.join(out_dir, "rgb_depth_side.mp4")
    tp_path = os.path.join(out_dir, "third_person.mp4")
    print(f"[INFO] Output: {out_dir}")
    print(f"[INFO] bin={bin_xy.tolist()} waypoints={waypoints.tolist()}")
    print(
        f"[INFO] third-person eye={np.round(tp_eye, 2).tolist()} "
        f"look_at={np.round(tp_target, 2).tolist()}"
    )
    print(
        "[INFO] depth proxies: "
        + ("on (hide for RGB, show for depth)" if use_depth_proxies else "off")
    )

    load_lab_scene_usd()

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
    checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)

    spawn_yaw = float(np.arctan2(waypoints[1, 1] - waypoints[0, 1], waypoints[1, 0] - waypoints[0, 0]))
    spawn_quat = yaw_to_quat_wxyz(spawn_yaw)

    env_cfg = G1FlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 100000
    env_cfg.curriculum = None
    env_cfg.scene.robot.init_state.rot = spawn_quat
    env_cfg.decimation = 1
    env_cfg.sim.render_interval = 1
    env_cfg.terminations.base_contact = None
    env_cfg.commands.base_velocity.debug_vis = False

    add_blue_bin(env_cfg, pos=(float(bin_xy[0]), float(bin_xy[1]), 0.5), index=0)

    cam_cfg = build_rtx_camera_cfg(
        img_res=img_res,
        update_period_s=0.0,
        sim_utils=sim_utils,
        camera_cfg_cls=CameraCfg,
    )
    cam_cfg.data_types = ["rgb", "distance_to_image_plane"]
    cam_cfg.depth_clipping_behavior = "max"
    env_cfg.scene.camera = cam_cfg
    env_cfg.scene.tp_camera = build_tp_camera_cfg(
        eye=tp_eye,
        target=tp_target,
        height=int(args_cli.tp_height_px),
        width=int(args_cli.tp_width_px),
        sim_utils_mod=sim_utils,
        camera_cfg_cls=CameraCfg,
    )
    print("[INFO] Head FPV + world third-person CameraCfg.")

    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
    device = env.unwrapped.device
    sim_dt = float(env.unwrapped.cfg.sim.dt)
    scene = env.unwrapped.scene

    sim = env.unwrapped.sim
    if sim.has_gui():
        sim.set_render_mode(sim.RenderMode.PARTIAL_RENDERING)
        print("[INFO] GUI open: PARTIAL_RENDERING.")
    if hasattr(sim, "set_camera_view"):
        sim.set_camera_view(
            eye=[float(tp_eye[0]), float(tp_eye[1]), float(tp_eye[2])],
            target=[float(tp_target[0]), float(tp_target[1]), float(tp_target[2])],
        )

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath("/World/ground"):
        stage.RemovePrim("/World/ground")
        print("[INFO] Removed default /World/ground so it does not z-fight the lab floor.")
    if not args_cli.no_nurec_ccm and float(args_cli.nurec_deyellow) > 1e-6:
        apply_nurec_floor_color(deyellow=float(args_cli.nurec_deyellow), verbose=True)
    else:
        print("[INFO] NuRec floor CCM: off")

    add_bin_fill_light((float(bin_xy[0]), float(bin_xy[1])), intensity=12000.0)
    apply_blue_bin_visual("/World/envs/env_0/blue_bin_0")
    if use_depth_proxies:
        found = set_lab_depth_proxies_visible(stage, False)
        print(f"[INFO] Lab depth proxies hidden until depth pass: {found or 'none found'}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=device)

    robot = scene["robot"]
    num_joints = int(robot.data.joint_pos.shape[1])
    camera = scene["camera"]
    tp_camera = scene["tp_camera"]

    obs, _ = env.reset()
    spawn = torch.tensor(
        [[
            float(waypoints[0, 0]),
            float(waypoints[0, 1]),
            0.8,
            float(spawn_quat[0]),
            float(spawn_quat[1]),
            float(spawn_quat[2]),
            float(spawn_quat[3]),
        ]],
        device=device,
    )
    robot.write_root_pose_to_sim(spawn)
    if hasattr(robot, "write_root_velocity_to_sim"):
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

    bin_asset = scene["blue_bin_0"]
    bin_pose = torch.tensor(
        [[float(bin_xy[0]), float(bin_xy[1]), 0.5, 0.5, 0.5, 0.5, 0.5]],
        device=device,
    )
    bin_asset.write_root_pose_to_sim(bin_pose)
    if hasattr(bin_asset, "write_root_velocity_to_sim"):
        bin_asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

    commands = torch.zeros(1, 3, device=device)
    zero_actions = torch.zeros(1, num_joints, device=device)
    warmup = 30
    print(f"[INFO] Warmup {warmup} steps...")
    for _ in range(warmup):
        env.unwrapped.command_manager._terms["base_velocity"].command[:] = commands
        obs, _, _, _ = env.step(zero_actions)

    bin_asset.write_root_pose_to_sim(bin_pose)
    if hasattr(bin_asset, "write_root_velocity_to_sim"):
        bin_asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))
    bin_xyz = bin_asset.data.root_pos_w[0].detach().cpu().numpy()
    print(f"[INFO] blue_bin_0 at ({bin_xyz[0]:.2f},{bin_xyz[1]:.2f},{bin_xyz[2]:.2f})")

    nav = ShortestYawWander(
        max_speed=float(args_cli.max_speed),
        stop_thresh=float(args_cli.wp_stop_m),
    )
    wp_idx = 1
    target = waypoints[wp_idx].copy()
    print(f"[INFO] First target wp[{wp_idx}]={target}")

    num_steps = int(round(float(args_cli.duration_s) / sim_dt))
    capture_fps = max(float(args_cli.video_fps), 1e-3)
    playback_fps = max(float(args_cli.playback_fps), 1e-3)
    video_period_s = 1.0 / capture_fps
    slow_x = capture_fps / playback_fps
    next_frame_t = 0.0
    frame_idx = 0
    done_hold = 0
    side_writer = _open_video_writer(side_path, playback_fps)
    tp_writer = _open_video_writer(tp_path, playback_fps)

    print(
        f"[INFO] Traj max {num_steps} steps ({args_cli.duration_s:.1f}s sim, "
        f"dt={sim_dt:.4f}s, capture {capture_fps:.1f} fps, "
        f"playback {playback_fps:.1f} fps, {slow_x:.2f}x slower)"
    )

    try:
        for step in range(num_steps):
            data = robot.data
            base_pos = data.root_pos_w[0].detach().cpu().numpy()
            base_quat = data.root_quat_w[0].detach().cpu().numpy()

            cmd, dist, reached = nav.command(base_pos, base_quat, target)
            if reached:
                if wp_idx + 1 < len(waypoints):
                    wp_idx += 1
                    target = waypoints[wp_idx].copy()
                    print(f"[INFO] Reached waypoint. New target wp[{wp_idx}]={target}")
                    cmd, dist, _ = nav.command(base_pos, base_quat, target)
                else:
                    done_hold += 1
                    cmd = np.zeros(3, dtype=np.float32)
                    if done_hold >= int(round(0.6 / sim_dt)):
                        print("[INFO] Last waypoint reached; stopping.")
                        break

            commands[0, 0] = float(cmd[0])
            commands[0, 1] = float(cmd[1])
            commands[0, 2] = float(cmd[2])
            env.unwrapped.command_manager._terms["base_velocity"].command[:] = commands
            if "policy" in obs:
                obs["policy"][0, 11] = commands[0, 2]

            with torch.inference_mode():
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            sim_time_s = step * sim_dt
            if sim_time_s + 1e-12 >= next_frame_t:
                fpv_rgb = _read_with_retry(
                    lambda: _rgb_from_camera(camera, rotate=True),
                    env,
                    commands,
                    zero_actions,
                    "fpv rgb",
                )
                tp_rgb = _read_with_retry(
                    lambda: _rgb_from_camera(tp_camera, rotate=False),
                    env,
                    commands,
                    zero_actions,
                    "third-person rgb",
                )
                if use_depth_proxies:
                    set_lab_depth_proxies_visible(stage, True)
                    _refresh_camera(camera, sim)
                    depth_vis = _depth_from_camera(camera, max_depth)
                    set_lab_depth_proxies_visible(stage, False)
                else:
                    depth_vis = _depth_from_camera(camera, max_depth)
                side = _hstack_rgb(fpv_rgb, depth_vis)
                side_writer.append_data(side)
                tp_writer.append_data(tp_rgb)

                if args_cli.save_frames:
                    imageio.imwrite(os.path.join(frames_dir, f"side_{frame_idx:06d}.png"), side)
                    imageio.imwrite(os.path.join(frames_dir, f"tp_{frame_idx:06d}.png"), tp_rgb)
                if frame_idx == 0:
                    imageio.imwrite(os.path.join(out_dir, "preview_side.png"), side)
                    imageio.imwrite(os.path.join(out_dir, "preview_tp.png"), tp_rgb)
                    print(f"[INFO] Preview: {os.path.join(out_dir, 'preview_side.png')}")
                    print(f"[INFO] Preview TP: {os.path.join(out_dir, 'preview_tp.png')}")

                frame_idx += 1
                next_frame_t += video_period_s

            if step % 200 == 0:
                print(
                    f"[STEP {step}/{num_steps}] xy=({base_pos[0]:.2f},{base_pos[1]:.2f}) "
                    f"wp[{wp_idx}]={target} dist={dist:.2f} frames={frame_idx}"
                )
    finally:
        side_writer.close()
        tp_writer.close()

    print(f"[INFO] Wrote {frame_idx} frames")
    print(f"[INFO] FPV RGB|depth: {side_path}")
    print(f"[INFO] Third-person:  {tp_path}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
