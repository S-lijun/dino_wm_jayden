"""Test HJ safety filter on Isaac G1 latent humanoid (switching / waypoint / safe-only).

Mirrors CarGoal ``test_cargoal_latent_specific_checkpoint.py`` but:
  - nominal = WaypointNavController (not Dreamer)
  - loads training ``policy.pth`` (actor+critic inside)
  - critic-only gate: Q(z, a_nom) < threshold → SF takeover
  - no WM dynamics rollout (can add later)

Usage (env_isaaclab)::

  python test_HJ_humanoid.py --headless --visual_mode rtx_rgb \\
    --dino_ckpt_dir C:\\ --dino_encoder wm_ckpt_18-27-17 --with_proprio \\
    --policy_path runs/ddpg_hj_humanoid/.../epoch_id_N/policy.pth \\
    --mode switching --num_runs 5
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

parser = argparse.ArgumentParser("Test HJ safety filter on latent Humanoid (Isaac G1)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--dino_ckpt_dir",
    type=str,
    default="C:\\",
    help="Parent of encoder folder (e.g. C:\\ for C:\\wm_ckpt_18-27-17)",
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
    help="Path to training checkpoint .../epoch_id_N/policy.pth",
)
parser.add_argument(
    "--mode",
    type=str,
    default="switching",
    choices=["switching", "waypoint_only", "safe_only"],
    help="Controller mode (CarGoal: switching / pid_only / safe_only).",
)
parser.add_argument("--num_runs", type=int, default=5)
parser.add_argument(
    "--safety_threshold",
    type=float,
    default=0.0,
    help="Switch to SF when Q(z, a_nom) < this (default 0).",
)
parser.add_argument(
    "--max_visual_steps",
    type=int,
    default=400,
    help="Hard cap on Gym/HJ visual steps per episode.",
)
parser.add_argument(
    "--out_dir",
    type=str,
    default=None,
    help="Where to save videos/summary (default: humanoid_test/<timestamp>).",
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


def build_policy(env: LatentHumanoidEnv, args, device: str):
    """Rebuild avoid-DDPG policy to match training, then load policy.pth."""
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


@torch.no_grad()
def q_value(policy, z: np.ndarray, act_policy: np.ndarray, device: str) -> float:
    """Q(z, a) with a in policy space [-1, 1]."""
    z_t = torch.as_tensor(z, dtype=torch.float32, device=device)
    if z_t.ndim == 1:
        z_t = z_t.unsqueeze(0)
    a_t = torch.as_tensor(act_policy, dtype=torch.float32, device=device)
    if a_t.ndim == 1:
        a_t = a_t.unsqueeze(0)
    q = policy.critic(z_t, a_t)
    return float(q.reshape(-1)[0].item())


@torch.no_grad()
def safe_action_env(
    policy,
    z: np.ndarray,
    device: str,
    n_candidates: int = 64,
) -> np.ndarray:
    """Critic-greedy SF → env-space (vx, vy, yaw): max Q over sampled actions."""
    from env.isaac.critic_select import select_action_max_q_numpy

    act_pol, _ = select_action_max_q_numpy(
        policy.critic,
        z,
        act_dim=3,
        n_candidates=n_candidates,
        device=device,
    )
    return np.asarray(policy.map_action(act_pol), dtype=np.float32).reshape(3)


def simulate_one(
    env: LatentHumanoidEnv,
    policy,
    *,
    mode: str,
    device: str,
    safety_threshold: float,
    max_visual_steps: int,
    save_video: bool,
    out_dir: Path,
    run_id: int,
) -> dict:
    z, info = env.reset()
    frames: list[np.ndarray] = []
    if save_video:
        frames.append(_visual_to_uint8_hwc(env.wrapper.get_raw_obs()["visual"]))

    hj_interventions = 0
    total_switches = 0
    constraint_violations = 0
    min_hj_nom = float("inf")
    last_controller = None
    end_reason = None
    h_s_last = float("nan")

    for step in range(max_visual_steps):
        a_nom_env = env.compute_waypoint_nav_action()
        a_nom_pol = np.asarray(
            policy.map_action_inverse(a_nom_env), dtype=np.float32
        ).reshape(-1)

        if mode == "waypoint_only":
            action = a_nom_env
            using_hj = False
            q_nom = q_value(policy, z, a_nom_pol, device)
        elif mode == "safe_only":
            action = safe_action_env(policy, z, device)
            using_hj = True
            q_nom = q_value(policy, z, a_nom_pol, device)
        else:  # switching
            q_nom = q_value(policy, z, a_nom_pol, device)
            if q_nom < safety_threshold:
                action = safe_action_env(policy, z, device)
                using_hj = True
                hj_interventions += 1
                if last_controller == "waypoint":
                    total_switches += 1
                last_controller = "hj"
                if step < 30 or hj_interventions <= 5:
                    print(
                        f"  step {step}: SF intervene Q(a_nom)={q_nom:.3f} "
                        f"a_nom={a_nom_env} a_sf={action}"
                    )
            else:
                action = a_nom_env
                using_hj = False
                if last_controller == "hj":
                    total_switches += 1
                last_controller = "waypoint"

        min_hj_nom = min(min_hj_nom, q_nom)

        z, h_s, terminated, truncated, info = env.step(action)
        h_s_last = float(h_s)
        if h_s < 0:
            constraint_violations += 1

        if save_video:
            frames.append(_visual_to_uint8_hwc(env.wrapper.get_raw_obs()["visual"]))

        if terminated or truncated:
            # info["end_reason"] is int code
            code = int(info.get("end_reason", 0))
            end_reason = {
                0: "ongoing",
                1: "all_waypoints",
                2: "stuck",
                3: "max_steps",
                4: "out_of_bounds",
            }.get(code, f"code_{code}")
            break
    else:
        end_reason = "max_visual_steps"

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
        "min_q_nom": min_hj_nom if np.isfinite(min_hj_nom) else None,
        "final_h_s": h_s_last,
        "stuck": float(info.get("stuck", 0.0)),
        "video": video_path,
    }
    print(
        f"[RUN {run_id}] mode={mode} end={end_reason} success={success} "
        f"steps={result['visual_steps']} interventions={hj_interventions} "
        f"violations={constraint_violations} minQ={result['min_q_nom']}"
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
    out_dir = Path(args.out_dir) if args.out_dir else Path("humanoid_test") / stamp
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

    # No wandb video spam during closed-loop test.
    env = LatentHumanoidEnv(
        args,
        wm,
        args.device,
        args_cli,
        with_proprio=bool(args.with_proprio),
        latent_h=False,
        wandb_video_every=0,
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
                max_visual_steps=int(args.max_visual_steps),
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

    summary_path = out_dir / f"summary_{args.mode}.txt"
    lines = [
        f"mode={args.mode}",
        f"policy={args.policy_path}",
        f"runs={n}",
        f"success={n_ok}/{n} ({100.0 * n_ok / max(n, 1):.1f}%)",
        f"stuck={n_stuck}",
        f"out_of_bounds={n_oob}",
        f"mean_hj_interventions={mean_int:.2f}",
        f"mean_constraint_violations={mean_viol:.2f}",
        "",
        "per-run:",
    ]
    for r in results:
        lines.append(
            f"  run{r['run_id']:02d}: end={r['end_reason']} success={r['success']} "
            f"steps={r['visual_steps']} int={r['hj_interventions']} "
            f"viol={r['constraint_violations']} minQ={r['min_q_nom']}"
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
