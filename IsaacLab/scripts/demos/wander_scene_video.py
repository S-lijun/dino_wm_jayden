"""Wander the current lab USD scene and export a first-person RGB video.

Same stack as DataCollection / IsaacG1Wrapper (G1 + pretrained locomotion,
lab.usda, blue bin, head camera), but no CSV / LiDAR / contact dumps.
The robot keeps sampling random aisle waypoints so you can judge how close
the current mesh scene looks to the real lab — without Gaussian Splatting.

Usage (from IsaacLab/):

    # Windows, headless RTX RGB (default)
    .\\isaaclab.bat -p scripts/demos/wander_scene_video.py

    # Open Isaac GUI
    .\\isaaclab.bat -p scripts/demos/wander_scene_video.py --no_headless

    # Linux
    ./isaaclab.sh -p scripts/demos/wander_scene_video.py

    # Cluster-safe depth colormap (not photoreal)
    ./isaaclab.sh -p scripts/demos/wander_scene_video.py --headless --visual_mode depth_rgb

    # Shorter clip
    .\\isaaclab.bat -p scripts/demos/wander_scene_video.py --duration_s 20

    # No GS lab: Isaac default grid + default skyLight (compare lighting)
    .\\isaaclab.bat -p scripts/demos/wander_scene_video.py --no_lab_scene --duration_s 20

    # No GS lab, Isaac lights off (is the mesh bin still visible?)
    .\\isaaclab.bat -p scripts/demos/wander_scene_video.py --no_lab_scene --no_isaac_light --duration_s 20

    # Optional later grade (off by default; first wander was raw RTX)
    .\\isaaclab.bat -p scripts/demos/wander_scene_video.py --duration_s 20 --grade --exposure_ev -0.6
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
    description="Wander the current G1 lab scene and export an RGB video."
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
    "--visual_mode",
    type=str,
    default="rtx_rgb",
    choices=["rtx_rgb", "depth_rgb"],
    help="rtx_rgb = true USD/RTX RGB (realism check). depth_rgb = depth colormap.",
)
parser.add_argument("--duration_s", type=float, default=30.0, help="Sim time to wander (s).")
parser.add_argument("--video_fps", type=float, default=15.0, help="Exported video fps.")
parser.add_argument("--img_height", type=int, default=640, help="Portrait sensor height.")
parser.add_argument("--img_width", type=int, default=480, help="Portrait sensor width.")
parser.add_argument("--max_speed", type=float, default=0.45, help="Planar wander speed (m/s).")
parser.add_argument(
    "--bin_x",
    type=float,
    default=3.5,
    help="Blue bin world X (training aisle default).",
)
parser.add_argument(
    "--bin_y",
    type=float,
    default=-2.0,
    help="Blue bin world Y (training aisle default).",
)
parser.add_argument("--spawn_x", type=float, default=0.0, help="Robot spawn X.")
parser.add_argument("--spawn_y", type=float, default=-2.0, help="Robot spawn Y.")
parser.add_argument(
    "--wander_x_min", type=float, default=0.2, help="Random waypoint X min."
)
parser.add_argument(
    "--wander_x_max", type=float, default=4.2, help="Random waypoint X max."
)
parser.add_argument(
    "--wander_y_min", type=float, default=-3.5, help="Random waypoint Y min."
)
parser.add_argument(
    "--wander_y_max", type=float, default=-0.5, help="Random waypoint Y max."
)
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
    help="Output folder. Default: IsaacLab/data/wander_video/<timestamp>/",
)
parser.add_argument(
    "--save_frames",
    action="store_true",
    help="Also write rgb_XXXXXX.png next to the mp4.",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--no_lab_scene",
    action="store_true",
    help="Do not load lab.usda / 3DGS. Keep Isaac default ground grid.",
)
parser.add_argument(
    "--no_isaac_light",
    action="store_true",
    help="Disable Isaac skyLight and the local bin fill light.",
)
parser.add_argument(
    "--exposure_ev",
    type=float,
    default=-0.6,
    help="Only used with --grade. Negative darkens.",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=1.08,
    help="Only used with --grade.",
)
parser.add_argument(
    "--highlight",
    type=float,
    default=0.45,
    help="Only used with --grade.",
)
parser.add_argument(
    "--saturation",
    type=float,
    default=1.06,
    help="Only used with --grade.",
)
parser.add_argument(
    "--wb_temp",
    type=float,
    default=0.0,
    help="Camera RGB WB. Off by default; floor color is NuRec CCM.",
)
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
    "--contrast",
    type=float,
    default=0.0,
    help="Midtone S-curve. 0 = off. Try 0.20 if you want more punch.",
)
parser.add_argument(
    "--grade",
    action="store_true",
    help="Also apply the old darken grade. Off by default.",
)
args_cli = parser.parse_args()

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
from isaaclab.sensors.ray_caster import RayCasterCameraCfg, patterns
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import (
    G1FlatEnvCfg_PLAY,
)

from data_collection_obstacles import add_bin_fill_light, add_blue_bin
from lab_scene_utils import (
    apply_nurec_floor_color,
    default_raycast_mesh_paths,
    load_lab_scene_usd,
    rotate_sensor_ccw_to_landscape,
)
from visual_obs_utils import (
    build_depth_camera_cfgs,
    build_rtx_camera_cfg,
    depth_to_rgb,
    grade_rgb,
    merge_depth_maps_multi,
)

TASK = "Isaac-Velocity-Flat-G1-v0"
RL_LIBRARY = "rsl_rl"


def quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class ShortestYawWander:
    """Same shortest-yaw tracker as DataCollection / WaypointNavController."""

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


def _rgb_to_uint8(rgb_np: np.ndarray) -> np.ndarray:
    if rgb_np.dtype != np.uint8:
        rgb_np = (np.clip(rgb_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    return rgb_np


def _read_rtx_rgb(camera, env, commands, zero_actions) -> np.ndarray:
    last_exc: Exception | None = None
    for attempt in range(15):
        try:
            rgb_tensor = camera.data.output["rgb"][0]
            rgb_np = rgb_tensor[..., :3].detach().cpu().numpy()
            rgb_np = _rgb_to_uint8(rgb_np)
            return rotate_sensor_ccw_to_landscape(rgb_np)
        except TypeError as exc:
            last_exc = exc
            env.unwrapped.command_manager._terms["base_velocity"].command[:] = commands
            env.step(zero_actions)
            if attempt == 0 or attempt == 14:
                print(f"[WARN] rtx_rgb read failed (attempt {attempt + 1}/15): {exc}")
    raise RuntimeError(
        "rtx_rgb never produced a valid frame. Try GUI (no --headless) or --visual_mode depth_rgb."
    ) from last_exc


def _read_depth_rgb(scene, depth_cam_names: list[str]) -> np.ndarray:
    depth_list = [
        scene[n].data.output["distance_to_image_plane"][0].detach().cpu().numpy()
        for n in depth_cam_names
    ]
    merged = merge_depth_maps_multi(depth_list, max_d=10.0)
    return rotate_sensor_ccw_to_landscape(depth_to_rgb(merged))


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

    if args_cli.output_dir:
        out_dir = os.path.abspath(args_cli.output_dir)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../data/wander_video", stamp)
        )
    os.makedirs(out_dir, exist_ok=True)
    frames_dir = os.path.join(out_dir, "frames")
    if args_cli.save_frames:
        os.makedirs(frames_dir, exist_ok=True)
    video_path = os.path.join(out_dir, "wander.mp4")
    print(f"[INFO] Output: {out_dir}")
    print(f"[INFO] visual_mode={_visual_mode}")
    load_lab = not bool(args_cli.no_lab_scene)
    use_isaac_light = not bool(args_cli.no_isaac_light)
    print(f"[INFO] lab_scene={'off (Isaac default grid)' if not load_lab else 'on (GS lab.usda)'}")
    print(f"[INFO] isaac_light={'on' if use_isaac_light else 'off'}")
    print(f"[INFO] wb_temp={args_cli.wb_temp:g} (negative = yellow→grey, no blue boost)")
    print(f"[INFO] contrast={args_cli.contrast:g}")
    if args_cli.grade:
        print(
            f"[INFO] camera grade: exposure_ev={args_cli.exposure_ev:g} "
            f"gamma={args_cli.gamma:g} highlight={args_cli.highlight:g} "
            f"saturation={args_cli.saturation:g}"
        )
    else:
        print("[INFO] camera grade darken: off")

    # RTX: load lab before env so the camera annotator sees the composed stage.
    if load_lab and _visual_mode in ("rtx_rgb", "depth_rgb"):
        load_lab_scene_usd()

    depth_cam_names: list[str] = []
    if _visual_mode == "depth_rgb":
        mesh_paths = default_raycast_mesh_paths(
            "/World/envs/env_0",
            obstacle_names=("blue_bin_0",),
            include_lab_scene=load_lab,
        )
    else:
        mesh_paths = ()

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
    if not use_isaac_light:
        env_cfg.scene.sky_light = None

    add_blue_bin(env_cfg, pos=(float(bin_xy[0]), float(bin_xy[1]), 0.5), index=0)

    if _visual_mode == "rtx_rgb":
        env_cfg.scene.camera = build_rtx_camera_cfg(
            img_res=img_res,
            update_period_s=0.0,
            sim_utils=sim_utils,
            camera_cfg_cls=CameraCfg,
        )
        print("[INFO] rtx_rgb: head CameraCfg (USD/RTX, no 3DGS).")
    else:
        for name, dc_cfg in build_depth_camera_cfgs(
            mesh_paths,
            img_res=img_res,
            update_period_s=1.0 / float(args_cli.video_fps),
            patterns_mod=patterns,
            ray_caster_camera_cfg_cls=RayCasterCameraCfg,
        ):
            setattr(env_cfg.scene, name, dc_cfg)
            depth_cam_names.append(name)
        print(f"[INFO] depth_rgb: {len(depth_cam_names)} RayCasterCamera sensors.")

    env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=env_cfg))
    device = env.unwrapped.device
    sim_dt = float(env.unwrapped.cfg.sim.dt)
    scene = env.unwrapped.scene

    sim = env.unwrapped.sim
    if sim.has_gui() and _visual_mode == "rtx_rgb":
        sim.set_render_mode(sim.RenderMode.PARTIAL_RENDERING)
        print("[INFO] GUI open: PARTIAL_RENDERING (viewport stays usable).")

    # G1FlatEnvCfg respawns /World/ground after the pre-env lab load.
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if load_lab and stage.GetPrimAtPath("/World/ground"):
        stage.RemovePrim("/World/ground")
        print("[INFO] Removed default /World/ground so it does not z-fight the lab floor.")
    elif not load_lab:
        print("[INFO] Isaac default grid kept at /World/ground.")
    if load_lab and not args_cli.no_nurec_ccm and float(args_cli.nurec_deyellow) > 1e-6:
        apply_nurec_floor_color(deyellow=float(args_cli.nurec_deyellow), verbose=True)
    elif load_lab:
        print("[INFO] NuRec floor CCM: off")

    if not use_isaac_light:
        if stage.GetPrimAtPath("/World/skyLight"):
            stage.RemovePrim("/World/skyLight")
            print("[INFO] Removed /World/skyLight (Isaac outdoor HDR).")
        print("[INFO] Isaac daylight off.")
    else:
        print("[INFO] Isaac daylight /World/skyLight left on (mesh bin needs it).")
        if load_lab:
            add_bin_fill_light((float(args_cli.bin_x), float(args_cli.bin_y)))
    leftover_lights = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.IsValid() and "Light" in prim.GetTypeName()
    ]
    print(f"[INFO] USD lights: {leftover_lights or 'none'}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=device)

    robot = scene["robot"]
    num_joints = int(robot.data.joint_pos.shape[1])
    camera = scene["camera"] if _visual_mode == "rtx_rgb" else None

    obs, _ = env.reset()
    spawn = torch.tensor(
        [[float(args_cli.spawn_x), float(args_cli.spawn_y), 0.8, 1.0, 0.0, 0.0, 0.0]],
        device=device,
    )
    robot.write_root_pose_to_sim(spawn)
    if hasattr(robot, "write_root_velocity_to_sim"):
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device))

    commands = torch.zeros(1, 3, device=device)
    zero_actions = torch.zeros(1, num_joints, device=device)
    warmup = 30 if _visual_mode == "rtx_rgb" else 3
    print(f"[INFO] Warmup {warmup} steps...")
    for _ in range(warmup):
        env.unwrapped.command_manager._terms["base_velocity"].command[:] = commands
        obs, _, _, _ = env.step(zero_actions)

    nav = ShortestYawWander(max_speed=float(args_cli.max_speed))
    target = _sample_wander_xy(
        rng,
        bin_xy,
        float(args_cli.bin_keepout_m),
        float(args_cli.wander_x_min),
        float(args_cli.wander_x_max),
        float(args_cli.wander_y_min),
        float(args_cli.wander_y_max),
    )
    print(f"[INFO] First wander target: {target}")

    num_steps = int(round(float(args_cli.duration_s) / sim_dt))
    video_period_s = 1.0 / float(args_cli.video_fps)
    next_frame_t = 0.0
    frame_idx = 0
    writer = _open_video_writer(video_path, float(args_cli.video_fps))

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
                target = _sample_wander_xy(
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
                if _visual_mode == "rtx_rgb":
                    rgb = _read_rtx_rgb(camera, env, commands, zero_actions)
                else:
                    rgb = _read_depth_rgb(scene, depth_cam_names)
                apply_wb = abs(float(args_cli.wb_temp)) > 1e-6
                apply_contrast = abs(float(args_cli.contrast)) > 1e-6
                if apply_wb or apply_contrast or args_cli.grade:
                    rgb = grade_rgb(
                        rgb,
                        exposure_ev=float(args_cli.exposure_ev) if args_cli.grade else 0.0,
                        gamma=float(args_cli.gamma) if args_cli.grade else 1.0,
                        highlight=float(args_cli.highlight) if args_cli.grade else 0.0,
                        saturation=float(args_cli.saturation) if args_cli.grade else 1.0,
                        wb_temp=float(args_cli.wb_temp),
                        contrast=float(args_cli.contrast),
                    )
                writer.append_data(rgb)
                if args_cli.save_frames:
                    imageio.imwrite(
                        os.path.join(frames_dir, f"rgb_{frame_idx:06d}.png"),
                        rgb,
                    )
                if frame_idx == 0:
                    preview = os.path.join(out_dir, "preview.png")
                    imageio.imwrite(preview, rgb)
                    print(f"[INFO] Preview frame: {preview}")
                frame_idx += 1
                next_frame_t += video_period_s

            if step % 200 == 0:
                print(
                    f"[STEP {step}/{num_steps}] xy=({base_pos[0]:.2f},{base_pos[1]:.2f}) "
                    f"target={target} dist={dist:.2f} frames={frame_idx}"
                )
    finally:
        writer.close()

    print(f"[INFO] Wrote {frame_idx} frames → {video_path}")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
