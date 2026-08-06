"""Test QP safety filter on Isaac G1 latent humanoid.

Loads critic from ``train_HJ_humanoid_qp.py`` checkpoints. Control uses
WaypointNavController as nominal + QP filter (closest safe action to a_nom
with Q >= threshold). Default QP search is stratified continuous sampling
(one uniform draw per action-space bin). Yaw free by default.

Default ``--pass_side back`` puts the terminal goal past the bin on the
centerline (5.0, 0). Use ``middle`` to aim at the bin, or ``left_right`` for
the previous left|right eval.

Cannot reuse ``test_HJ_humanoid.py`` (actor switching / safe_only).

Usage (env_isaaclab)::

  python test_HJ_humanoid_qp.py --headless --visual_mode rtx_rgb \\
    --dino_ckpt_dir /workspace --dino_encoder wm_ckpt_18-27-17 --with_proprio \\
    --policy_path runs/qp_hj_humanoid/.../epoch_id_N/policy.pth \\
    --mode QP --num_runs 5 --pass_side back
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------
# Isaac Sim must start before isaaclab / env imports.
# ---------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
ISAACLAB_ROOT = os.path.join(REPO_ROOT, "IsaacLab")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, ISAACLAB_ROOT)

import scripts.reinforcement_learning.rsl_rl.cli_args as cli_args
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser("Test QP HJ filter on latent Humanoid (Isaac G1)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--dino_ckpt_dir",
    type=str,
    default="/workspace",
    help="Parent of encoder folder (e.g. /workspace for /workspace/wm_ckpt_18-27-17)",
)
parser.add_argument(
    "--dino_encoder",
    type=str,
    default="wm_ckpt_18-27-17",
    help="Encoder / WM run folder under dino_ckpt_dir",
)
parser.add_argument(
    "--config",
    type=str,
    default="train_HJ_configs.yaml",
    help="Same YAML as training (net sizes, etc.)",
)
parser.add_argument("--with_proprio", action="store_true")
parser.add_argument(
    "--visual_mode",
    type=str,
    default="rtx_rgb",
    choices=["off", "depth_rgb", "lidar_rgb", "rtx_rgb"],
)
parser.add_argument(
    "--policy_path",
    type=str,
    required=True,
    help="Path to QP training checkpoint .../epoch_id_N/policy.pth",
)
parser.add_argument(
    "--mode",
    type=str,
    default="QP",
    choices=["QP", "waypoint_only"],
    help="QP = stratified continuous filter on critic; waypoint_only = no filter.",
)
parser.add_argument("--num_runs", type=int, default=5)
parser.add_argument(
    "--safety_threshold",
    type=float,
    default=0.0,
    help="QP: intervene when Q(z, a_nom) < this (default 0).",
)
parser.add_argument(
    "--qp_n_grid",
    type=int,
    default=21,
    help=(
        "Bins per axis for QP search (21³ if yaw free; 21² if --freeze_yaw). "
        "Default stratified mode draws one continuous action uniformly in each bin."
    ),
)
parser.add_argument(
    "--qp_sample_mode",
    type=str,
    default="stratified",
    choices=["stratified", "fixed"],
    help=(
        "stratified: one continuous uniform sample per axis-aligned bin (default). "
        "fixed: legacy linspace lattice (actions stick to discrete levels)."
    ),
)
parser.add_argument(
    "--no_vy_yaw_same_sign",
    action="store_true",
    help="Disable QP constraint that candidates must have sign(vy)==sign(yaw).",
)
parser.add_argument(
    "--freeze_yaw",
    action="store_true",
    help="Freeze yaw=0 in a_nom / QP grid (legacy). Default: yaw free.",
)
parser.add_argument(
    "--max_visual_steps",
    type=int,
    default=400,
    help="Hard cap on Gym/HJ visual steps per episode.",
)
parser.add_argument(
    "--goal_radius",
    type=float,
    default=0.05,
    help="Reach goal when dist(robot, goal_xy) <= this (meters). Also stops robot.",
)
parser.add_argument(
    "--pass_side",
    type=str,
    default="back",
    choices=["back", "middle", "left", "right", "left_right"],
    help=(
        "Terminal goal after front. "
        "back = past bin on centerline (5.0,0); "
        "middle = aim at bin (3.5,0); "
        "left_right = cycle left|right only."
    ),
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=None,
    help="Where to save videos/summary (default: humanoid_test_qp/<timestamp>).",
)
parser.add_argument(
    "--save_video",
    action="store_true",
    default=True,
    help="Save mp4 for each run (default on).",
)
parser.add_argument(
    "--no_save_video",
    action="store_true",
    help="Disable video saving.",
)

args_cli, remaining = parser.parse_known_args()

sys.path.insert(0, os.path.join(ISAACLAB_ROOT, "scripts/demos"))
from visual_obs_utils import configure_app_for_visual, resolve_visual_mode

_visual_mode = resolve_visual_mode(args_cli)
configure_app_for_visual(args_cli, _visual_mode)
torch_device = getattr(args_cli, "device", "cuda:0")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------
# Imports after AppLauncher
# ---------------------------------------------------------------------
import gym
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

from PyHJ.exploration import GaussianNoise
from PyHJ.policy import avoid_DDPGPolicy_annealing
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic

from wm_load import load_model
from env.isaac.latent_humanoid_env import LatentHumanoidEnv
from env.isaac.qp_filter import qp_filter_action_policy


def args_type(default):
    def parse_string(x):
        if default is None:
            return x
        if isinstance(default, bool):
            return bool(["False", "True"].index(x))
        if isinstance(default, int):
            return float(x) if ("e" in x or "." in x) else int(x)
        if isinstance(default, (list, tuple)):
            return tuple(args_type(default[0])(y) for y in x.split(","))
        return type(default)(x)

    def parse_object(x):
        if isinstance(default, (list, tuple)):
            return tuple(x)
        return x

    return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


def get_args_and_merge_config():
    with open(args_cli.config) as f:
        cfg = yaml.safe_load(f)

    cfg_parser = argparse.ArgumentParser()
    for key, val in sorted(cfg.items()):
        arg_t = args_type(val)
        cfg_parser.add_argument(f"--{key}", type=arg_t, default=arg_t(val))
    cfg_args = cfg_parser.parse_args(remaining)

    args = argparse.Namespace(**vars(args_cli))
    for key, val in vars(cfg_args).items():
        setattr(args, key.replace("-", "_"), val)
    return args


def _visual_to_uint8_hwc(visual) -> np.ndarray:
    if isinstance(visual, torch.Tensor):
        arr = visual.detach().cpu().numpy()
    else:
        arr = np.asarray(visual)
    if arr.ndim != 3:
        raise ValueError(f"Expected HxWxC visual, got {arr.shape}")
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


def _video_frame(env: LatentHumanoidEnv, hj_val: float | None, l_val: float | None) -> np.ndarray:
    """Full-res visual with top-left HJ / l overlay."""
    hwc = _visual_to_uint8_hwc(env.wrapper.get_raw_obs()["visual"])
    return env._overlay_metrics_hwc(hwc, hj_val, l_val)


def _goal_xy(env: LatentHumanoidEnv) -> np.ndarray:
    """Terminal goal = last waypoint (left/right of bin)."""
    wps = np.asarray(env.wrapper.waypoints, dtype=np.float64)
    if wps.ndim == 1:
        return wps.reshape(2)
    return wps[-1].reshape(2)


def _robot_xy(env: LatentHumanoidEnv) -> np.ndarray:
    return np.asarray(env.wrapper.get_robot_xy_local(), dtype=np.float64).reshape(2)


def _obstacle_xy(env: LatentHumanoidEnv) -> np.ndarray:
    xy = getattr(env.wrapper, "_blue_bin_xy", (3.5, 0.0))
    return np.asarray([float(xy[0]), float(xy[1])], dtype=np.float64)


def _dist_to_goal(env: LatentHumanoidEnv) -> float:
    r = _robot_xy(env)
    g = _goal_xy(env)
    return float(np.hypot(r[0] - g[0], r[1] - g[1]))


def _save_topdown_traj(
    *,
    out_path: Path,
    traj: np.ndarray,
    obstacle_xy: np.ndarray,
    goal_xy: np.ndarray,
    start_xy: np.ndarray | None = None,
    goal_radius: float = 0.05,
    title: str = "",
) -> None:
    """Top-down x–y plot: blue obstacle, goal, robot trajectory → PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    traj = np.asarray(traj, dtype=np.float64).reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    if traj.shape[0] > 0:
        ax.plot(traj[:, 0], traj[:, 1], "-", color="0.25", linewidth=1.5, label="robot traj")
        ax.scatter(traj[0, 0], traj[0, 1], c="green", s=40, zorder=3, label="start")
        ax.scatter(traj[-1, 0], traj[-1, 1], c="orange", s=40, zorder=3, label="end")
    if start_xy is not None:
        ax.scatter(
            float(start_xy[0]), float(start_xy[1]), c="green", s=40, zorder=3, marker="o"
        )
    ax.scatter(
        float(obstacle_xy[0]),
        float(obstacle_xy[1]),
        c="blue",
        s=80,
        zorder=4,
        label="obstacle",
    )
    ax.add_patch(
        Circle(
            (float(goal_xy[0]), float(goal_xy[1])),
            float(goal_radius),
            fill=False,
            edgecolor="red",
            linestyle="--",
            linewidth=1.2,
            label=f"goal r={goal_radius:.2f}",
        )
    )
    ax.scatter(
        float(goal_xy[0]), float(goal_xy[1]), c="red", s=50, marker="*", zorder=4, label="goal"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[INFO] Saved top-down traj: {out_path}")


def build_policy(env: LatentHumanoidEnv, args, device: str):
    """Rebuild avoid-DDPG shell to load policy.pth (critic used; actor unused)."""
    state_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    max_action = 1.0
    policy_action_space = gym.spaces.Box(
        low=np.asarray(env.action_space.low, dtype=np.float32),
        high=np.asarray(env.action_space.high, dtype=np.float32),
        dtype=np.float32,
    )

    critic_net = Net(
        state_shape,
        action_shape,
        hidden_sizes=args.critic_net,
        activation=getattr(torch.nn, args.critic_activation),
        concat=True,
        device=device,
    )
    critic = Critic(critic_net, device=device).to(device)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=float(args.critic_lr))

    actor_net = Net(
        state_shape,
        hidden_sizes=args.control_net,
        activation=getattr(torch.nn, args.actor_activation),
        device=device,
    )
    actor = Actor(
        actor_net, action_shape, max_action=max_action, device=device
    ).to(device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=float(args.actor_lr))

    policy = avoid_DDPGPolicy_annealing(
        critic=critic,
        critic_optim=critic_optim,
        tau=float(args.tau),
        gamma=float(args.gamma_pyhj),
        exploration_noise=GaussianNoise(sigma=0.0),
        reward_normalization=bool(args.rew_norm),
        estimation_step=int(args.n_step),
        action_space=policy_action_space,
        actor=actor,
        actor_optim=actor_optim,
        actor_gradient_steps=int(args.actor_gradient_steps),
    )
    policy.new_expl = False
    return policy


def load_policy_checkpoint(policy, ckpt_path: str | Path, device: str) -> None:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"--policy_path not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(state, strict=True)
    policy.actor.eval()
    policy.critic.eval()
    print(f"[INFO] Loaded policy from {ckpt_path}")


def simulate_one(
    env: LatentHumanoidEnv,
    policy,
    *,
    mode: str,
    device: str,
    safety_threshold: float,
    qp_n_grid: int,
    qp_sample_mode: str,
    require_vy_yaw_same_sign: bool,
    freeze_yaw: bool,
    max_visual_steps: int,
    goal_radius: float,
    save_video: bool,
    out_dir: Path,
    run_id: int,
) -> dict:
    z, info = env.reset()
    frames: list[np.ndarray] = []
    h_s_last = float(env.wrapper.calculate_cost())
    hj_exec = env._hj_value(z)
    if save_video:
        frames.append(_video_frame(env, hj_exec, h_s_last))

    goal_xy = _goal_xy(env)
    obstacle_xy = _obstacle_xy(env)
    start_xy = _robot_xy(env)
    traj: list[np.ndarray] = [start_xy.copy()]
    zero_action = np.zeros(3, dtype=np.float32)

    hj_interventions = 0
    total_switches = 0
    constraint_violations = 0
    fallback_maxq = 0
    min_hj_nom = float("inf")
    last_controller = None
    end_reason = None
    step = -1

    print(
        f"[INFO] goal={goal_xy} obstacle={obstacle_xy} "
        f"goal_radius={goal_radius:.3f} start={start_xy}"
    )

    for step in range(max_visual_steps):
        # Already inside goal ball → force stop and end (do not let QP keep pushing).
        if _dist_to_goal(env) <= goal_radius:
            action = zero_action
            z, h_s, terminated, truncated, info = env.step(action)
            h_s_last = float(h_s)
            traj.append(_robot_xy(env))
            if save_video:
                hj_exec = env._hj_value(z, action_env=action.astype(np.float64))
                frames.append(_video_frame(env, hj_exec, h_s_last))
            end_reason = "all_waypoints"
            print(
                f"  step {step}: GOAL reached dist={_dist_to_goal(env):.3f} "
                f"<= {goal_radius:.3f}; stop action=[0,0,0]"
            )
            break

        a_nom_env = env.compute_waypoint_nav_action()
        a_nom_pol = np.asarray(
            policy.map_action_inverse(a_nom_env), dtype=np.float32
        ).reshape(-1)
        if freeze_yaw and a_nom_pol.size >= 3:
            a_nom_pol[2] = 0.0
        a_nom_env_use = np.asarray(policy.map_action(a_nom_pol), dtype=np.float32).reshape(3)
        if freeze_yaw:
            a_nom_env_use[2] = 0.0

        if mode == "waypoint_only":
            action = a_nom_env_use
            q_nom = float("nan")
            with torch.no_grad():
                z_t = torch.as_tensor(z, dtype=torch.float32, device=device)
                if z_t.ndim == 1:
                    z_t = z_t.unsqueeze(0)
                a_t = torch.as_tensor(a_nom_pol, dtype=torch.float32, device=device).unsqueeze(0)
                q_nom = float(policy.critic(z_t, a_t).reshape(-1)[0].item())
        else:  # QP
            act_pol, qp_info = qp_filter_action_policy(
                policy.critic,
                z,
                a_nom_pol,
                safety_threshold=safety_threshold,
                n_grid=qp_n_grid,
                freeze_yaw=freeze_yaw,
                yaw=0.0,
                device=device,
                sample_mode=qp_sample_mode,  # type: ignore[arg-type]
                require_vy_yaw_same_sign=require_vy_yaw_same_sign,
            )
            q_nom = float(qp_info["q_nom"])
            q_qp = qp_info.get("q_chosen")
            if q_qp is not None:
                q_qp = float(q_qp)
            using_hj = bool(qp_info["intervened"])
            if qp_info.get("fallback_maxq"):
                fallback_maxq += 1
            if using_hj:
                action = np.asarray(policy.map_action(act_pol), dtype=np.float32).reshape(3)
                if freeze_yaw:
                    action[2] = 0.0
                # If filter omitted q_chosen, evaluate critic on chosen policy action.
                if q_qp is None:
                    with torch.no_grad():
                        z_t = torch.as_tensor(z, dtype=torch.float32, device=device)
                        if z_t.ndim == 1:
                            z_t = z_t.unsqueeze(0)
                        a_t = torch.as_tensor(
                            act_pol, dtype=torch.float32, device=device
                        ).unsqueeze(0)
                        q_qp = float(policy.critic(z_t, a_t).reshape(-1)[0].item())
                hj_interventions += 1
                if last_controller == "waypoint":
                    total_switches += 1
                last_controller = "qp"
                # Always log fallbacks; also early steps / first few interventions.
                if (
                    step < 30
                    or hj_interventions <= 5
                    or qp_info.get("fallback_maxq")
                    or step % 20 == 0
                ):
                    print(
                        f"  step {step}: QP intervene "
                        f"Q(a_nom)={q_nom:.6f} Q(a_qp)={q_qp:.6f} "
                        f"Δ={float(q_qp) - q_nom:+.6f} "
                        f"Q[min,max,std]="
                        f"[{qp_info.get('q_min'):.6f},"
                        f"{qp_info.get('q_max'):.6f},"
                        f"{qp_info.get('q_std'):.6f}] "
                        f"n_cand={qp_info.get('n_candidates')} "
                        f"n_safe={qp_info['n_safe']} "
                        f"fallback={qp_info.get('fallback_maxq')} "
                        f"a_nom={a_nom_env_use} a_qp={action}"
                    )
            else:
                action = a_nom_env_use
                if last_controller == "qp":
                    total_switches += 1
                last_controller = "waypoint"

        min_hj_nom = min(min_hj_nom, q_nom)

        z, h_s, terminated, truncated, info = env.step(action)
        h_s_last = float(h_s)
        traj.append(_robot_xy(env))
        if h_s < 0:
            constraint_violations += 1

        if save_video:
            hj_exec = env._hj_value(
                z, action_env=np.asarray(action, dtype=np.float64)
            )
            frames.append(_video_frame(env, hj_exec, h_s_last))

        # Reached goal this step → settle with zero cmd then end.
        if _dist_to_goal(env) <= goal_radius:
            z, h_s, _, _, info = env.step(zero_action)
            h_s_last = float(h_s)
            traj.append(_robot_xy(env))
            if save_video:
                hj_exec = env._hj_value(z, action_env=zero_action.astype(np.float64))
                frames.append(_video_frame(env, hj_exec, h_s_last))
            end_reason = "all_waypoints"
            print(
                f"  step {step}: GOAL reached dist={_dist_to_goal(env):.3f}; "
                f"stop action=[0,0,0]"
            )
            break

        if terminated or truncated:
            code = int(info.get("end_reason", 0))
            end_reason = {
                0: "ongoing",
                1: "all_waypoints",
                2: "stuck",
                3: "max_steps",
                4: "out_of_bounds",
            }.get(code, f"code_{code}")
            # Env reported goal — still send a stop for video / residual velocity.
            if end_reason == "all_waypoints":
                z, h_s, _, _, info = env.step(zero_action)
                h_s_last = float(h_s)
                traj.append(_robot_xy(env))
                if save_video:
                    hj_exec = env._hj_value(z, action_env=zero_action.astype(np.float64))
                    frames.append(_video_frame(env, hj_exec, h_s_last))
            break
    else:
        end_reason = "max_visual_steps"

    traj_arr = np.stack(traj, axis=0) if traj else np.zeros((0, 2))
    traj_path = out_dir / f"{mode}_run{run_id:02d}_traj.png"
    _save_topdown_traj(
        out_path=traj_path,
        traj=traj_arr,
        obstacle_xy=obstacle_xy,
        goal_xy=goal_xy,
        start_xy=start_xy,
        goal_radius=goal_radius,
        title=f"{mode} run{run_id:02d} end={end_reason}",
    )

    video_path = None
    if save_video and frames:
        try:
            import imageio.v2 as imageio

            video_path = str(out_dir / f"{mode}_run{run_id:02d}.mp4")
            imageio.mimsave(video_path, frames, fps=int(round(env.visual_fps)))
            print(f"[INFO] Saved video: {video_path} ({len(frames)} frames)")
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] video save failed: {exc}")

    success = end_reason == "all_waypoints"
    result = {
        "mode": mode,
        "run_id": run_id,
        "end_reason": end_reason,
        "success": success,
        "visual_steps": int(info.get("episode_step", step + 1)),
        "hj_interventions": hj_interventions,
        "total_switches": total_switches,
        "constraint_violations": constraint_violations,
        "fallback_maxq": fallback_maxq,
        "min_q_nom": min_hj_nom if np.isfinite(min_hj_nom) else None,
        "final_h_s": h_s_last,
        "final_dist_goal": _dist_to_goal(env),
        "stuck": float(info.get("stuck", 0.0)),
        "video": video_path,
        "traj_png": str(traj_path),
    }
    print(
        f"[RUN {run_id}] mode={mode} end={end_reason} success={success} "
        f"steps={result['visual_steps']} interventions={hj_interventions} "
        f"violations={constraint_violations} fallback_maxq={fallback_maxq} "
        f"minQ={result['min_q_nom']} dist_goal={result['final_dist_goal']:.3f}"
    )
    return result


def main():
    args = get_args_and_merge_config()
    args.device = torch_device
    args.dino_ckpt_dir = os.path.join(args.dino_ckpt_dir, args.dino_encoder)
    save_video = bool(args.save_video) and not bool(args.no_save_video)

    np.random.seed(int(getattr(args, "seed", 0)))
    torch.manual_seed(int(getattr(args, "seed", 0)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(getattr(args, "seed", 0)))

    stamp = datetime.now().strftime("%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("humanoid_test_qp") / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] out_dir={out_dir}")
    print(f"[INFO] mode={args.mode} policy={args.policy_path}")

    ckpt_dir = Path(args.dino_ckpt_dir)
    hydra_cfg = ckpt_dir / "hydra.yaml"
    snapshot = ckpt_dir / "checkpoints" / "model_latest.pth"
    train_cfg = OmegaConf.load(str(hydra_cfg))
    wm = load_model(snapshot, train_cfg, train_cfg.num_action_repeat, device=args.device)
    wm.eval()
    for p in wm.parameters():
        p.requires_grad = False

    env = LatentHumanoidEnv(
        args,
        wm,
        args.device,
        args_cli,
        with_proprio=bool(args.with_proprio),
        latent_h=False,
        wandb_video_every=0,
    )
    # Terminal goal after front. Default back = past bin on centerline (5.0, 0).
    pass_side = str(getattr(args, "pass_side", "back"))
    if pass_side == "left_right":
        env.wrapper.include_middle_pass = False
        # Keep default sequence tuple → cycles/samples left|right only.
    else:
        env.wrapper.include_middle_pass = pass_side == "middle"
        env.wrapper.trajectory_region_sequence = ["start", "front", pass_side]
        if pass_side == "back" and "back" not in env.wrapper.trajectory_regions:
            env.wrapper.trajectory_regions["back"] = {
                "mode": "point",
                "xy": (5.0, 0.0),
            }
    env.wrapper.y_bound = 0.0  # disable |y| corridor
    goal_radius = float(args.goal_radius)
    env.wrapper.waypoint_stop_thresh = goal_radius
    env.waypoint_nav.stop_thresh = goal_radius
    back_xy = env.wrapper.trajectory_regions.get("back", {}).get("xy", (5.0, 0.0))
    print(
        f"[INFO] pass_side={pass_side} "
        f"sequence={env.wrapper.trajectory_region_sequence} "
        f"regions.back={back_xy} "
        f"goal_radius={goal_radius} (env stop_thresh synced)"
    )
    if pass_side == "back":
        print(
            f"[INFO] Terminal goal should be behind obstacle at {tuple(back_xy)} "
            f"(bin stays at {tuple(env.wrapper._blue_bin_xy_fixed)})"
        )
    policy = build_policy(env, args, args.device)
    load_policy_checkpoint(policy, args.policy_path, args.device)
    env.policy_for_log = policy

    results = []
    for run_id in range(int(args.num_runs)):
        print(f"\n=== Run {run_id + 1}/{args.num_runs} ({args.mode}) ===")
        results.append(
            simulate_one(
                env,
                policy,
                mode=args.mode,
                device=args.device,
                safety_threshold=float(args.safety_threshold),
                qp_n_grid=int(args.qp_n_grid),
                qp_sample_mode=str(getattr(args, "qp_sample_mode", "stratified")),
                require_vy_yaw_same_sign=not bool(
                    getattr(args, "no_vy_yaw_same_sign", False)
                ),
                freeze_yaw=bool(args.freeze_yaw),
                max_visual_steps=int(args.max_visual_steps),
                goal_radius=goal_radius,
                save_video=save_video,
                out_dir=out_dir,
                run_id=run_id,
            )
        )

    n = len(results)
    n_ok = sum(1 for r in results if r["success"])
    n_stuck = sum(1 for r in results if r["end_reason"] == "stuck")
    n_oob = sum(1 for r in results if r["end_reason"] == "out_of_bounds")
    mean_int = float(np.mean([r["hj_interventions"] for r in results])) if n else 0.0
    mean_viol = float(np.mean([r["constraint_violations"] for r in results])) if n else 0.0
    mean_fb = float(np.mean([r["fallback_maxq"] for r in results])) if n else 0.0

    summary_path = out_dir / f"summary_{args.mode}.txt"
    lines = [
        f"mode={args.mode}",
        f"policy={args.policy_path}",
        f"runs={n}",
        f"success={n_ok}/{n} ({100.0 * n_ok / max(n, 1):.1f}%)",
        f"stuck={n_stuck}",
        f"out_of_bounds={n_oob}",
        f"mean_qp_interventions={mean_int:.2f}",
        f"mean_constraint_violations={mean_viol:.2f}",
        f"mean_fallback_maxq={mean_fb:.2f}",
        f"goal_radius={float(args.goal_radius)}",
        f"pass_side={getattr(args, 'pass_side', 'back')}",
        "",
        "per-run:",
    ]
    for r in results:
        lines.append(
            f"  run{r['run_id']:02d}: end={r['end_reason']} success={r['success']} "
            f"steps={r['visual_steps']} int={r['hj_interventions']} "
            f"viol={r['constraint_violations']} fallback={r['fallback_maxq']} "
            f"minQ={r['min_q_nom']} dist_goal={r.get('final_dist_goal')} "
            f"traj={r.get('traj_png')}"
        )
    text = "\n".join(lines) + "\n"
    summary_path.write_text(text, encoding="utf-8")
    print("\n" + "=" * 60)
    print(text)
    print(f"[INFO] Wrote {summary_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
