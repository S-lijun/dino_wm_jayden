"""Collect RTX camera RGB + official ``distance_to_image_plane`` depth.

Same stack as wander_scene_video / IsaacG1Wrapper (G1 + pretrained locomotion,
lab.usda 3DGS, blue bin, head CameraCfg). Depth is the USD/RTX annotator, not
RayCasterCamera / LiDAR range-image.

3DGS/NuRec has no mesh, so RTX depth would miss the floor unless the lab
collision proxies (GroundPlane + wall planes) are visible. Default is a
two-pass capture: RGB with proxies hidden (GS carpet), then one extra render
with proxies visible for depth.

Usage (from IsaacLab/):

    .\\isaaclab.bat -p scripts/demos/collect_rtx_depth.py --headless --duration_s 20

    .\\isaaclab.bat -p scripts/demos/collect_rtx_depth.py --no_headless --duration_s 20

    ./isaaclab.sh -p scripts/demos/collect_rtx_depth.py --headless --duration_s 20
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
    description="Wander the GS lab and dump RTX RGB + distance_to_image_plane depth."
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
parser.add_argument("--duration_s", type=float, default=20.0, help="Sim time to wander (s).")
parser.add_argument("--video_fps", type=float, default=15.0, help="Exported video fps.")
parser.add_argument("--img_height", type=int, default=640, help="Portrait sensor height.")
parser.add_argument("--img_width", type=int, default=480, help="Portrait sensor width.")
parser.add_argument("--max_speed", type=float, default=0.45, help="Planar wander speed (m/s).")
parser.add_argument("--max_depth", type=float, default=10.0, help="Colormap clip (meters).")
parser.add_argument("--bin_x", type=float, default=3.5, help="Blue bin world X.")
parser.add_argument("--bin_y", type=float, default=-2.0, help="Blue bin world Y.")
parser.add_argument("--spawn_x", type=float, default=0.0, help="Robot spawn X.")
parser.add_argument("--spawn_y", type=float, default=-2.0, help="Robot spawn Y.")
parser.add_argument("--wander_x_min", type=float, default=0.2)
parser.add_argument("--wander_x_max", type=float, default=4.2)
parser.add_argument("--wander_y_min", type=float, default=-3.5)
parser.add_argument("--wander_y_max", type=float, default=-0.5)
parser.add_argument(
    "--bin_keepout_m",
    type=float,
    default=0.7,
    help="Do not sample wander targets inside this radius of the bin.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="Output folder. Default: IsaacLab/data/rtx_depth/<timestamp>/",
)
parser.add_argument(
    "--save_frames",
    action="store_true",
    help="Also write rgb_XXXXXX.png and depth_XXXXXX.png.",
)
parser.add_argument(
    "--save_raw_depth",
    action="store_true",
    help="Write float32 meters npy next to each depth frame (large).",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--nurec_deyellow",
    type=float,
    default=0.55,
    help="Session-only NuRec Volume CCM: yellow→grey. 0 = off.",
)
parser.add_argument(
    "--no_nurec_ccm",
    action="store_true",
    help="Do not override NuRec floor color.",
)
parser.add_argument(
    "--no_depth_proxies",
    action="store_true",
    help="Do not unhide lab collision meshes for depth (floor stays empty).",
)
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
)
from visual_obs_utils import build_rtx_camera_cfg

TASK = "Isaac-Velocity-Flat-G1-v0"
RL_LIBRARY = "rsl_rl"


def quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class ShortestYawWander:
    """Same shortest-yaw tracker as DataCollection / wander_scene_video."""

    def __init__(self, max_speed: float = 0.45, stop_thresh: float = 0.18):
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


def _clip_xy(xy: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> np.ndarray:
    return np.array(
        [float(np.clip(xy[0], x_min, x_max)), float(np.clip(xy[1], y_min, y_max))],
        dtype=np.float64,
    )


def _sample_wander_xy(
    rng: np.random.Generator,
    bin_xy: np.ndarray,
    keepout: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    for _ in range(64):
        xy = np.array(
            [rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)],
            dtype=np.float64,
        )
        if float(np.linalg.norm(xy - bin_xy)) >= keepout:
            return xy
    return np.array([x_min, 0.5 * (y_min + y_max)], dtype=np.float64)


def _sample_bin_view_xy(
    rng: np.random.Generator,
    bin_xy: np.ndarray,
    keepout: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    """Stand in the aisle looking at the bin so it stays in the head camera."""
    for _ in range(64):
        dist = float(rng.uniform(max(keepout + 0.15, 1.0), 2.2))
        ang = float(rng.uniform(-0.7, 0.7))
        xy = bin_xy + np.array([-dist * np.cos(ang), dist * np.sin(ang)], dtype=np.float64)
        xy = _clip_xy(xy, x_min, x_max, y_min, y_max)
        if float(np.linalg.norm(xy - bin_xy)) >= keepout:
            return xy
    return _sample_wander_xy(rng, bin_xy, keepout, x_min, x_max, y_min, y_max)


def _next_target(
    rng: np.random.Generator,
    bin_xy: np.ndarray,
    keepout: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> np.ndarray:
    if rng.random() < 0.7:
        return _sample_bin_view_xy(rng, bin_xy, keepout, x_min, x_max, y_min, y_max)
    return _sample_wander_xy(rng, bin_xy, keepout, x_min, x_max, y_min, y_max)


def _rgb_to_uint8(rgb_np: np.ndarray) -> np.ndarray:
    if rgb_np.dtype != np.uint8:
        rgb_np = (np.clip(rgb_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgb_np


# lab.usda collision proxies. Hidden for GS RGB; must be visible for RTX depth.
LAB_DEPTH_PROXY_PATHS = (
    "/World/ExternalScene/GroundPlane",
    "/World/ExternalScene/Plane",
    "/World/ExternalScene/plane2",
    "/World/ExternalScene/plane3",
    "/World/ExternalScene/plane4",
)


def _squeeze_hw(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr)
    while out.ndim > 2 and out.shape[-1] == 1:
        out = out[..., 0]
    if out.ndim != 2:
        raise ValueError(f"Expected HxW depth, got {out.shape}")
    return out


def _depth_to_gray_rgb(depth: np.ndarray, max_depth: float) -> np.ndarray:
    """Meters → 8-bit grayscale RGB (near=black, far/clip=white)."""
    d = np.asarray(depth, dtype=np.float32)
    d = np.nan_to_num(d, nan=max_depth, posinf=max_depth, neginf=0.0)
    d = np.clip(d, 0.0, max_depth)
    gray = np.round(d / max_depth * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _lab_depth_proxy_prims(stage):
    prims = []
    for path in LAB_DEPTH_PROXY_PATHS:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prims.append(prim)
    return prims


def _set_lab_depth_proxies_visible(stage, visible: bool) -> list[str]:
    from pxr import UsdGeom

    found: list[str] = []
    for prim in _lab_depth_proxy_prims(stage):
        img = UsdGeom.Imageable(prim)
        if visible:
            img.MakeVisible()
        else:
            img.MakeInvisible()
        found.append(str(prim.GetPath()))
    return found


def _refresh_camera(camera, sim, n_renders: int = 2) -> None:
    """Extra RTX frames, then force the Camera annotator to re-read."""
    for _ in range(n_renders):
        sim.render()
    if hasattr(camera, "_is_outdated"):
        camera._is_outdated[:] = True
    camera.update(0.0, force_recompute=True)


def _rgb_from_camera(camera) -> np.ndarray:
    rgb_tensor = camera.data.output["rgb"][0]
    rgb_np = _rgb_to_uint8(rgb_tensor[..., :3].detach().cpu().numpy())
    return rotate_sensor_ccw_to_landscape(rgb_np)


def _depth_from_camera(camera, max_depth: float) -> tuple[np.ndarray, np.ndarray]:
    depth_tensor = camera.data.output["distance_to_image_plane"][0]
    depth_m = _squeeze_hw(depth_tensor.detach().cpu().numpy()).astype(np.float32)
    depth_m = rotate_sensor_ccw_to_landscape(depth_m)
    return depth_m, _depth_to_gray_rgb(depth_m, max_depth)


def _read_rtx_rgb_depth(
    camera,
    env,
    sim,
    stage,
    commands,
    zero_actions,
    max_depth: float,
    *,
    use_depth_proxies: bool,
):
    last_exc: Exception | None = None
    rgb_np = None
    for attempt in range(15):
        try:
            rgb_np = _rgb_from_camera(camera)
            break
        except TypeError as exc:
            last_exc = exc
            env.unwrapped.command_manager._terms["base_velocity"].command[:] = commands
            env.step(zero_actions)
            if attempt == 0 or attempt == 14:
                print(f"[WARN] rtx rgb read failed (attempt {attempt + 1}/15): {exc}")
    if rgb_np is None:
        raise RuntimeError(
            "RTX camera never produced rgb. Try GUI (no --headless)."
        ) from last_exc

    if use_depth_proxies:
        found = _set_lab_depth_proxies_visible(stage, True)
        if not found:
            print("[WARN] No lab depth proxy prims; floor depth will stay empty.")
        try:
            _refresh_camera(camera, sim)
            depth_m, depth_vis = _depth_from_camera(camera, max_depth)
        finally:
            _set_lab_depth_proxies_visible(stage, False)
    else:
        depth_m, depth_vis = _depth_from_camera(camera, max_depth)
    return rgb_np, depth_m, depth_vis


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
    rng = np.random.default_rng(int(args_cli.seed))
    img_res = (int(args_cli.img_height), int(args_cli.img_width))
    bin_xy = np.array([float(args_cli.bin_x), float(args_cli.bin_y)], dtype=np.float64)
    max_depth = float(args_cli.max_depth)

    if args_cli.output_dir:
        out_dir = os.path.abspath(args_cli.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/rtx_depth", stamp)
        )
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    if args_cli.save_frames or args_cli.save_raw_depth:
        os.makedirs(frames_dir, exist_ok=True)

    rgb_path = os.path.join(out_dir, "rgb.mp4")
    depth_path = os.path.join(out_dir, "depth.mp4")
    side_path = os.path.join(out_dir, "rgb_depth_side.mp4")
    print(f"[INFO] Output: {out_dir}")
    print("[INFO] RTX CameraCfg data_types=['rgb', 'distance_to_image_plane']")
    print("[INFO] lab_scene=on (GS lab.usda), blue_bin on")
    use_depth_proxies = not bool(args_cli.no_depth_proxies)
    print(
        "[INFO] depth proxies: "
        + ("on (hide for RGB, show for depth)" if use_depth_proxies else "off")
    )

    load_lab_scene_usd()

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(TASK, args_cli)
    checkpoint = get_published_pretrained_checkpoint(RL_LIBRARY, TASK)

    env_cfg = G1FlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 100000
    env_cfg.curriculum = None
    env_cfg.scene.robot.init_state.rot = (0.0, 0.0, 0.0, 1.0)
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
    print("[INFO] Head camera: USD/RTX pinhole, official distance_to_image_plane.")

    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
    device = env.unwrapped.device
    sim_dt = float(env.unwrapped.cfg.sim.dt)
    scene = env.unwrapped.scene

    sim = env.unwrapped.sim
    if sim.has_gui():
        sim.set_render_mode(sim.RenderMode.PARTIAL_RENDERING)
        print("[INFO] GUI open: PARTIAL_RENDERING.")

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage.GetPrimAtPath("/World/ground"):
        stage.RemovePrim("/World/ground")
        print("[INFO] Removed default /World/ground so it does not z-fight the lab floor.")
    if not args_cli.no_nurec_ccm and float(args_cli.nurec_deyellow) > 1e-6:
        apply_nurec_floor_color(deyellow=float(args_cli.nurec_deyellow), verbose=True)
    else:
        print("[INFO] NuRec floor CCM: off")

    add_bin_fill_light((float(args_cli.bin_x), float(args_cli.bin_y)), intensity=12000.0)
    apply_blue_bin_visual("/World/envs/env_0/blue_bin_0")

    proxy_prims = [str(p.GetPath()) for p in _lab_depth_proxy_prims(stage)]
    if use_depth_proxies:
        _set_lab_depth_proxies_visible(stage, False)
        print(f"[INFO] Lab depth proxies (hidden until depth pass): {proxy_prims or 'none found'}")
    else:
        print(f"[INFO] Lab depth proxies left invisible: {proxy_prims or 'none found'}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=device)

    robot = scene["robot"]
    num_joints = int(robot.data.joint_pos.shape[1])
    camera = scene["camera"]
    print(f"[INFO] Camera annotators: {list(camera.data.output.keys())}")

    obs, _ = env.reset()
    spawn = torch.tensor(
        [[float(args_cli.spawn_x), float(args_cli.spawn_y), 0.8, 1.0, 0.0, 0.0, 0.0]],
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

    nav = ShortestYawWander(max_speed=float(args_cli.max_speed))
    target = np.array(
        [float(bin_xy[0]) - 1.1, float(bin_xy[1])],
        dtype=np.float64,
    )
    print(f"[INFO] First wander target (in front of bin): {target}")

    num_steps = int(round(float(args_cli.duration_s) / sim_dt))
    video_period_s = 1.0 / float(args_cli.video_fps)
    next_frame_t = 0.0
    frame_idx = 0
    rgb_writer = _open_video_writer(rgb_path, float(args_cli.video_fps))
    depth_writer = _open_video_writer(depth_path, float(args_cli.video_fps))
    side_writer = _open_video_writer(side_path, float(args_cli.video_fps))

    print(
        f"[INFO] Wandering {num_steps} steps ({args_cli.duration_s:.1f}s sim, "
        f"dt={sim_dt:.4f}s, video {args_cli.video_fps:.1f} fps)"
    )

    try:
        for step in range(num_steps):
            data = robot.data
            base_pos = data.root_pos_w[0].detach().cpu().numpy()
            base_quat = data.root_quat_w[0].detach().cpu().numpy()

            cmd, dist, reached = nav.command(base_pos, base_quat, target)
            if reached:
                target = _next_target(
                    rng,
                    bin_xy,
                    float(args_cli.bin_keepout_m),
                    float(args_cli.wander_x_min),
                    float(args_cli.wander_x_max),
                    float(args_cli.wander_y_min),
                    float(args_cli.wander_y_max),
                )
                print(f"[INFO] Reached waypoint. New target={target}")
                cmd, dist, _ = nav.command(base_pos, base_quat, target)

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
                rgb, depth_m, depth_vis = _read_rtx_rgb_depth(
                    camera,
                    env,
                    sim,
                    stage,
                    commands,
                    zero_actions,
                    max_depth,
                    use_depth_proxies=use_depth_proxies,
                )
                side = _hstack_rgb(rgb, depth_vis)
                rgb_writer.append_data(rgb)
                depth_writer.append_data(depth_vis)
                side_writer.append_data(side)

                if args_cli.save_frames:
                    imageio.imwrite(os.path.join(frames_dir, f"rgb_{frame_idx:06d}.png"), rgb)
                    imageio.imwrite(
                        os.path.join(frames_dir, f"depth_{frame_idx:06d}.png"), depth_vis
                    )
                if args_cli.save_raw_depth:
                    np.save(
                        os.path.join(frames_dir, f"depth_m_{frame_idx:06d}.npy"),
                        depth_m,
                    )

                if frame_idx == 0:
                    imageio.imwrite(os.path.join(out_dir, "preview_rgb.png"), rgb)
                    imageio.imwrite(os.path.join(out_dir, "preview_depth.png"), depth_vis)
                    imageio.imwrite(os.path.join(out_dir, "preview_side.png"), side)
                    finite = np.isfinite(depth_m)
                    dmin = float(depth_m[finite].min()) if finite.any() else float("nan")
                    dmax = float(depth_m[finite].max()) if finite.any() else float("nan")
                    print(
                        f"[INFO] Preview + first depth meters: "
                        f"shape={depth_m.shape} min={dmin:.3f} max={dmax:.3f}"
                    )
                    print(f"[INFO] Preview: {os.path.join(out_dir, 'preview_side.png')}")

                frame_idx += 1
                next_frame_t += video_period_s

            if step % 200 == 0:
                print(
                    f"[STEP {step}/{num_steps}] xy=({base_pos[0]:.2f},{base_pos[1]:.2f}) "
                    f"target={target} dist={dist:.2f} frames={frame_idx}"
                )
    finally:
        rgb_writer.close()
        depth_writer.close()
        side_writer.close()

    print(f"[INFO] Wrote {frame_idx} frames")
    print(f"[INFO] RGB:  {rgb_path}")
    print(f"[INFO] Depth grayscale: {depth_path}")
    print(f"[INFO] Side-by-side: {side_path}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
