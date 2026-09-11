"""SAC HJ safety-filter training on Isaac G1 latent space (start-goal path layout).

Separate from ``train_HJ_humanoid_sac.py`` (aisle / hemisphere / left-right).
Do not use this file as a drop-in replacement for the original pipeline.

Layout:
  each episode samples start + goal inside the arena rectangle
  x in [-1, 6], y in [-4, 2]; two transition waypoints offset ±2.5 m
  perpendicular to the start-goal segment, ordered near-start then near-goal.
  Hitting the rectangle edge resets; that transition is not stored for updates.

  h_s = lidar_min - 1.5 (LiDAR only; contact force is not folded in).
  Action box: vx [0,1], vy [-0.5,0.5], yaw [-1,1].
  Blue-bin placement is still the original fixed (3.5, -2) until specified.

  Default critic is LateFusionCritic: z and a each through an MLP to the
  same width, then concat and a Q head (patch-token z, ~73k-D).
  Optional ``--visual_feature cls --critic_fusion concat``: DINOv2 CLS (384)
  concatenated with the 3-D action at the first Linear. Default flags
  keep the late-fusion patch pipeline unchanged.
"""

import argparse
import os
import sys
import time
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

parser = argparse.ArgumentParser("SAC HJ on DINO latent Humanoid (start-goal path)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--dino_ckpt_dir",
    type=str,
    default="/workspace",
    help="Parent dir of the WM run folder (joins with --dino_encoder)",
)
parser.add_argument(
    "--config",
    type=str,
    default="train_HJ_configs.yaml",
    help="Path to flat YAML of hyperparameters",
)
parser.add_argument(
    "--with_proprio",
    action="store_true",
    help="Include proprioceptive embeddings in latent state",
)
parser.add_argument(
    "--dino_encoder",
    type=str,
    default="wm_ckpt_18-27-17",
    help="Encoder / WM run folder under dino_ckpt_dir",
)
parser.add_argument(
    "--visual_feature",
    type=str,
    default="patch",
    choices=["patch", "cls"],
    help=(
        "Visual tokens in z. 'patch' = all DINOv2 patch tokens (default). "
        "'cls' = DINOv2 CLS only (384-D); does not retrain the WM."
    ),
)
parser.add_argument(
    "--critic_fusion",
    type=str,
    default="late",
    choices=["late", "concat"],
    help=(
        "'late': encode z and a separately then fuse (default). "
        "'concat': cat(z, a) into the first Linear (use with CLS)."
    ),
)
parser.add_argument("--latent_h", default=False, action="store_true")
parser.add_argument(
    "--visual_mode",
    type=str,
    default="depth_rgb",
    choices=["off", "depth_rgb", "lidar_rgb", "rtx_rgb"],
)
parser.add_argument(
    "--resume_policy",
    type=str,
    default=None,
    help="Path to .../epoch_id_N/policy.pth to resume.",
)
parser.add_argument(
    "--wandb_video_every",
    type=int,
    default=1,
    help="Upload a rollout video to wandb every N finished episodes (0 disables).",
)
parser.add_argument(
    "--y_bound",
    type=float,
    default=0.0,
    help="Legacy |y - y_center| half-width. Path pipeline uses --use_arena_bounds.",
)
parser.add_argument(
    "--y_center",
    type=float,
    default=-2.0,
    help="Unused when arena bounds are on.",
)
parser.add_argument(
    "--x_bound_max",
    type=float,
    default=100.0,
    help="Legacy far-x wall. Disabled in practice when --use_arena_bounds.",
)
parser.add_argument(
    "--use_arena_bounds",
    action="store_true",
    help="Enable rectangular arena reset: x in [arena_x_min, arena_x_max], y in [arena_y_min, arena_y_max].",
)
parser.add_argument("--arena_x_min", type=float, default=-1.0)
parser.add_argument("--arena_x_max", type=float, default=6.0)
parser.add_argument("--arena_y_min", type=float, default=-4.0)
parser.add_argument("--arena_y_max", type=float, default=2.0)
parser.add_argument(
    "--skip_arena_oob_from_buffer",
    action="store_true",
    default=True,
    help="Do not store / train on the arena-edge reset transition (default on).",
)
parser.add_argument(
    "--no_skip_arena_oob_from_buffer",
    action="store_true",
    help="Store arena-OOB transitions in the replay buffer.",
)
parser.add_argument(
    "--lidar_distance_threshold",
    type=float,
    default=1.5,
    help="h_s = lidar_min - this (meters).",
)
parser.add_argument(
    "--lidar_h_half_fov_deg",
    type=float,
    default=60.0,
    help=(
        "Half-width (deg) of the forward cone used for l/h_s. "
        "Inside ±this, l is the true min range; outside, l=2.0. "
        "Default 60 → 120° FOV (camera-visible). Aisle pipeline keeps 90."
    ),
)
parser.add_argument(
    "--include_contact_in_hs",
    action="store_true",
    default=False,
    help=(
        "Fold non-foot obstacle contact into l/h_s (failure label). "
        "Default off: h_s = lidar_min - threshold only."
    ),
)
parser.add_argument(
    "--no_include_contact_in_hs",
    action="store_true",
    help="Supervise l from LiDAR only (ignore contact force). Default.",
)
parser.add_argument(
    "--contact_hs",
    type=float,
    default=-2.5,
    help="l/h_s cap when a non-foot link is in contact (must be < 0).",
)
parser.add_argument(
    "--max_episode_steps",
    type=int,
    default=20000,
    help=(
        "Sim-step cap per episode (dt=0.005 → seconds/5e-3). "
        "Path layout is longer than the aisle; default 20000 ≈ 100s. "
        "Original aisle pipeline keeps 8000."
    ),
)
parser.add_argument(
    "--yaw_limit",
    type=float,
    default=1.0,
    help="Abs yaw-rate action limit (env space). Original pipeline uses 0.5.",
)
parser.add_argument(
    "--waypoint_layout",
    type=str,
    default="start_goal_perp",
    choices=["regions", "start_goal", "start_goal_perp"],
    help=(
        "start_goal: straight start→goal, bins in the ±perp_offset band. "
        "start_goal_perp: start + 2 vias in that same band + goal."
    ),
)
parser.add_argument(
    "--perp_offset",
    type=float,
    default=2.5,
    help=(
        "Outer half-width (m) of the obstacle band around start–goal. "
        "Bin |offset| ~ U[--perp_keepout, this]. start_goal_perp vias still "
        "use U[-this, this]."
    ),
)
parser.add_argument(
    "--perp_keepout",
    type=float,
    default=0.25,
    help=(
        "Inner keep-out (m) on both sides of the start–goal line. Bins are "
        "not placed in ±this corridor (no dead-ahead obstacles). 0 = old "
        "full band. Default 0.25."
    ),
)
parser.add_argument(
    "--min_start_goal_dist",
    type=float,
    default=4.0,
    help="Reject start/goal samples closer than this (m).",
)
parser.add_argument(
    "--z_fuse_frames",
    type=int,
    default=5,
    help=(
        "Average the last N full z vectors (visual + proprio) and use that "
        "as critic/actor/switch input. 1 = original single-frame z. "
        "Default 5 = past 4 frames + current."
    ),
)
parser.add_argument(
    "--layout_seed",
    type=int,
    default=None,
    help=(
        "Seed only for waypoint/obstacle sampling. Default: fresh each process "
        "so two bash runs differ. Train --seed still applies to torch. "
        "Pass this to reproduce a layout sequence."
    ),
)
parser.add_argument(
    "--max_n_obstacles",
    type=int,
    default=2,
    help="Max bins spawned in the Isaac scene (need 2 for dual-LiDAR). Original pipeline omits this (1).",
)
parser.add_argument(
    "--path_obstacle_layout",
    action="store_true",
    default=True,
    help="Place bins on the start-goal perpendicular (±perp_offset). Default on for this pipeline.",
)
parser.add_argument(
    "--obstacle_absent_prob",
    type=float,
    default=0.1,
    help="P(no obstacles this episode).",
)
parser.add_argument(
    "--two_obstacle_prob",
    type=float,
    default=0.5,
    help="Given obstacles are present, P(two bins); else one. Max two.",
)
parser.add_argument(
    "--alternate_collect",
    action="store_true",
    default=True,
    help="Formal collect: even episodes waypoint-only, odd episodes Q-gate switch. Default on.",
)
parser.add_argument(
    "--no_alternate_collect",
    action="store_true",
    help="Disable waypoint/switch episode alternation (pure SF collect).",
)
parser.add_argument(
    "--randomize_nom_speed",
    action="store_true",
    help=(
        "Each episode samples controller |v| once from "
        "Uniform[--nom_speed_min, --nom_speed_max] and holds it. a_nom only. "
        "Off: fixed max_speed (0.5)."
    ),
)
parser.add_argument(
    "--nom_speed_min",
    type=float,
    default=0.05,
    help="Lower bound of Uniform |v| when --randomize_nom_speed is on.",
)
parser.add_argument(
    "--nom_speed_max",
    type=float,
    default=0.95,
    help="Upper bound of Uniform |v| when --randomize_nom_speed is on.",
)
parser.add_argument(
    "--nom_speed_mean",
    type=float,
    default=0.5,
    help="Gaussian mean of |v| when --randomize_nom_speed is on.",
)
parser.add_argument(
    "--nom_speed_std",
    type=float,
    default=0.3,
    help="Gaussian std of |v| when --randomize_nom_speed is on.",
)
parser.add_argument(
    "--nom_act_noise",
    action="store_true",
    help=(
        "After a_nom is computed (including |v| resample), add noise to "
        "(vx, vy, yaw) then clip. vx is independent; vy and yaw share one "
        "draw so their signs match. a_nom only. Width/σ is chosen once "
        "per episode from --nom_act_noise_stds (else --nom_act_noise_std). "
        "Distribution is --nom_act_noise_dist."
    ),
)
parser.add_argument(
    "--nom_act_noise_dist",
    type=str,
    default="uniform",
    choices=["uniform", "gaussian"],
    help=(
        "a_nom (vx,vy,yaw) noise. vx is independent; vy and yaw share one "
        "draw. uniform: vx += U[-w,w], ε~U[-w,w] → vy+=ε, yaw+=ε. "
        "gaussian: same with N(0,σ)."
    ),
)
parser.add_argument(
    "--nom_act_noise_std",
    type=float,
    default=0.2,
    help=(
        "Fixed half-width w (uniform: U[-w,w]) or σ (gaussian) if "
        "--nom_act_noise_stds is empty. Example: 0.2 → vx += U[-0.2,0.2], "
        "then one ε~U[-0.2,0.2] added to both vy and yaw."
    ),
)
parser.add_argument(
    "--nom_act_noise_stds",
    type=str,
    default="",
    help=(
        "Optional comma-separated episode-wise w/σ for a_nom (vx,vy,yaw). "
        "Each episode picks one uniformly, then every step uses that width. "
        "Empty string: fall back to --nom_act_noise_std. a_sf never."
    ),
)
parser.add_argument(
    "--buffer_warmup_steps",
    type=int,
    default=4000,
    help=(
        "Waypoint collect steps before critic/SAC updates. Path episodes "
        "are long; default 4000 so warmup covers several layouts. "
        "Aisle pipeline still uses 1000."
    ),
)
parser.add_argument(
    "--critic_warmup_updates",
    type=int,
    default=4000,
    help="Critic-only updates after buffer warm-up before joint SAC training.",
)
parser.add_argument(
    "--actor_bc_warmup_updates",
    type=int,
    default=0,
    help="Optional BC of actor mean to waypoint acts before joint SAC. Default 0.",
)
parser.add_argument(
    "--action_reg_coef",
    type=float,
    default=0.0,
    help=(
        "λ for MSE(a_sf, a_good) in policy space whenever HJ(Q(a_nom))<0, "
        "whether the env ran waypoint or the safety filter. HJ>=0: no "
        "regularizer that step. a_good is the perp-sidestep action computed "
        "on the first HJ<0 of a stretch and frozen until HJ>=0. Default 0 "
        "(disabled)."
    ),
)
parser.add_argument(
    "--a_good_hold_steps",
    type=int,
    default=0,
    help=(
        "<=0 (default): freeze a_good until HJ>=0. >0: legacy refresh of G "
        "every this many consecutive HJ<0 steps."
    ),
)
parser.add_argument(
    "--boundary_reg_coef",
    type=float,
    default=0.5,
    help="Penalty when |act| > 0.8. Set 0 to disable.",
)
parser.add_argument(
    "--freeze_yaw",
    action="store_true",
    help=(
        "Train/collect with yaw forced to 0 (SF + a_nom); only vx/vy are used. "
        "Buffer warm-up still stores full waypoint yaw. Default: off (yaw free)."
    ),
)
parser.add_argument(
    "--force_right_pass",
    action="store_true",
    help=(
        "Force every episode's pass-side waypoint to the right region "
        "(buffer + formal train). Default off: keep left|right(|middle) cycle."
    ),
)
parser.add_argument(
    "--spawn_hemisphere_pass",
    action="store_true",
    help=(
        "Formal train only: alternate spawn half-disks about (1.5,-2) and couple "
        "pass-side (left half→left region, right half→right). Avoids cross-aisle "
        "goals. Buffer always stays start→front→left|right|middle. Default off."
    ),
)
parser.add_argument(
    "--switch_collect",
    action="store_true",
    help=(
        "Formal-train collect: execute waypoint if min(Q1,Q2)(z, a_nom) >= "
        "--switch_threshold, else execute the safety filter. SF action only "
        "hits the env when the gate says unsafe. Actor still updates every "
        "learn step; λ_nom is kept: MSE(a_sf, a_nom) when Q(a_nom) >= threshold, "
        "MSE(a_sf, a_good) when Q(a_nom) < threshold. a_good is the waypoint "
        "action toward the closer fully-safe side point (3.5,-0.5)|(3.5,-3.5). "
        "Buffer warm-up stays waypoint-only. "
        "Default off."
    ),
)
parser.add_argument(
    "--switch_threshold",
    type=float,
    default=0.0,
    help="Q-gate threshold for --switch_collect. SF interacts iff Q(a_nom) < this.",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.2,
    help="SAC entropy temperature (ignored if --auto_alpha).",
)
parser.add_argument(
    "--auto_alpha",
    action="store_true",
    default=True,
    help="Auto-tune alpha (default on).",
)
parser.add_argument(
    "--no_auto_alpha",
    action="store_true",
    help="Disable auto alpha; use fixed --alpha.",
)
parser.add_argument(
    "--alpha_lr",
    type=float,
    default=3e-4,
    help="Learning rate for log_alpha when auto-tuning.",
)

args_cli, remaining = parser.parse_known_args()

sys.path.insert(0, os.path.join(ISAACLAB_ROOT, "scripts/demos"))
from visual_obs_utils import configure_app_for_visual, resolve_visual_mode

_visual_mode = resolve_visual_mode(args_cli)
configure_app_for_visual(args_cli, _visual_mode)
torch_device = getattr(args_cli, "device", "cuda:0")
if not bool(getattr(args_cli, "headless", False)):
    print(
        "[WARN] Running WITHOUT --headless. Prefer:\n"
        "  python train_HJ_humanoid_sac.py --headless --visual_mode rtx_rgb ..."
    )
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import yaml
import wandb
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from PyHJ.data import Batch, Collector, VectorReplayBuffer
from PyHJ.trainer import offpolicy_trainer
from PyHJ.trainer.offpolicy import OffpolicyTrainer
from PyHJ.env import DummyVectorEnv
from PyHJ.utils import WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import ActorProb
from PyHJ.policy import avoid_SACPolicy_annealing

_PROGRESS_BAR_LOSS_KEYS = ("loss/actor", "loss/critic1", "loss/critic2")


def _log_update_data_bar_filter(self, data, losses):
    for k in losses.keys():
        self.stat[k].add(losses[k])
        losses[k] = self.stat[k].get()
        if k in _PROGRESS_BAR_LOSS_KEYS:
            data[k] = f"{losses[k]:.3f}"
    self.logger.log_update_data(losses, self.gradient_step)


OffpolicyTrainer.log_update_data = _log_update_data_bar_filter  # type: ignore[method-assign]

from wm_load import load_model
from env.isaac.latent_humanoid_env import ENV_VY_RANGE, LatentHumanoidEnv
from env.isaac.skip_update_buffer import SkipUpdateReplayBuffer
from env.isaac.ckpt_utils import save_epoch_checkpoint
from env.isaac.late_fusion_critic import make_q_critic


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


def _nom_act_noise_banner(args, *, sep: str = ", ") -> str:
    if not bool(getattr(args, "nom_act_noise", False)):
        return ""
    dist = str(getattr(args, "nom_act_noise_dist", "uniform")).lower()
    stds = str(getattr(args, "nom_act_noise_stds", "") or "").strip()
    w = float(args.nom_act_noise_std)
    if dist == "uniform":
        if stds:
            return f"nom_act_U[-w,w] w∈{{{stds}}}/ep vx indep, vy=yaw{sep}"
        return f"nom_act_U[-{w:g},{w:g}] vx indep, vy=yaw{sep}"
    if stds:
        return f"nom_act_σ∈{{{stds}}}/ep vx indep, vy=yaw{sep}"
    return f"nom_act_N(0,{w:g}) vx indep, vy=yaw{sep}"


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


def _parse_resume_epoch_from_path(policy_path: str | Path) -> int:
    import re

    for part in Path(policy_path).resolve().parts[::-1]:
        m = re.fullmatch(r"epoch_id_(\d+)", part)
        if m:
            return int(m.group(1))
    raise ValueError(
        f"Cannot parse epoch from --resume_policy path (expected .../epoch_id_N/policy.pth): "
        f"{policy_path}"
    )


def load_policy_checkpoint(policy, ckpt_path: str | Path, device: str) -> None:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"--resume_policy not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    try:
        policy.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load checkpoint: critic architecture mismatch. "
            "Late-fusion (z_mlp/a_mlp) and concat (preprocess cat(z,a)) "
            "weights are not interchangeable; also match --visual_feature "
            f"(patch vs cls). Path: {ckpt_path}"
        ) from exc
    print(f"[INFO] Resumed policy from {ckpt_path}")


def main():
    args = get_args_and_merge_config()

    args.critic_lr = float(args.critic_lr)
    args.actor_lr = float(args.actor_lr)
    args.tau = float(args.tau)
    args.gamma_pyhj = float(args.gamma_pyhj)
    args.exploration_noise = float(args.exploration_noise)
    args.update_per_step = float(args.update_per_step)
    args.step_per_epoch = int(args.step_per_epoch)
    args.step_per_collect = int(args.step_per_collect)
    args.test_num = int(args.test_num)
    args.training_num = int(args.training_num)
    args.total_episodes = int(args.total_episodes)
    args.batch_size_pyhj = int(args.batch_size_pyhj)
    args.buffer_size = int(args.buffer_size)
    args.dino_ckpt_dir = os.path.join(args.dino_ckpt_dir, args.dino_encoder)
    args.device = torch_device
    args.visual_feature = str(getattr(args, "visual_feature", "patch")).lower()
    args.critic_fusion = str(getattr(args, "critic_fusion", "late")).lower()
    if args.visual_feature == "patch" and args.critic_fusion == "concat":
        print(
            "[WARN] critic_fusion=concat with patch z (~73k-D) will hide the "
            "3-D action; use --critic_fusion late or --visual_feature cls."
        )
    if args.visual_feature == "cls" and args.critic_fusion == "late":
        print(
            "[INFO] CLS z with late fusion; pass --critic_fusion concat to "
            "cat(CLS, action) at the first Linear."
        )
    use_auto_alpha = bool(args.auto_alpha) and not bool(args.no_auto_alpha)
    if bool(getattr(args, "no_skip_arena_oob_from_buffer", False)):
        args.skip_arena_oob_from_buffer = False
    else:
        args.skip_arena_oob_from_buffer = True
    if bool(getattr(args, "no_include_contact_in_hs", False)):
        args.include_contact_in_hs = False
    elif not bool(getattr(args, "include_contact_in_hs", False)):
        args.include_contact_in_hs = False
    if bool(getattr(args, "no_alternate_collect", False)):
        args.alternate_collect = False
    else:
        args.alternate_collect = True

    if args.training_num > 1:
        print("[WARN] Isaac Sim supports one env instance; forcing training_num=1")
        args.training_num = 1
    if args.test_num > 1:
        args.test_num = 1

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    from datetime import datetime

    timestamp = datetime.now().strftime("%m%d_%H%M")
    run_tag = "sac-path"
    if args.visual_feature == "cls" or args.critic_fusion == "concat":
        run_tag = f"sac-path-{args.visual_feature}-{args.critic_fusion}"
    resume_epoch_for_name = 0
    if args.resume_policy:
        resume_epoch_for_name = _parse_resume_epoch_from_path(args.resume_policy)
    run_name = f"{run_tag}-{args.dino_encoder}-{timestamp}"
    if args.resume_policy:
        run_name = f"{run_name}-resume{resume_epoch_for_name}"
    wandb.init(
        project="sac-hj-latent-humanoid",
        name=run_name,
        config=vars(args),
    )
    wandb.define_metric("trainer/env_step")
    wandb.define_metric("trainer/update")
    wandb.define_metric("safety/*", step_metric="trainer/env_step")
    wandb.define_metric("actor_action/*", step_metric="trainer/env_step")
    wandb.define_metric("rollout/*", step_metric="trainer/env_step")
    wandb.define_metric("loss/*", step_metric="trainer/env_step")
    wandb.define_metric("train/*", step_metric="trainer/env_step")
    wandb.define_metric("switch/*", step_metric="trainer/env_step")
    tb_dir = f"runs/sac_hj_humanoid_path/{run_name}/logs"
    writer = SummaryWriter(log_dir=tb_dir)
    wb_logger = WandbLogger(update_interval=1, train_interval=10**9, test_interval=10**9)
    wb_logger.load(writer)
    logger = wb_logger

    ckpt_dir = Path(args.dino_ckpt_dir)
    hydra_cfg = ckpt_dir / "hydra.yaml"
    snapshot = ckpt_dir / "checkpoints" / "model_latest.pth"
    if not hydra_cfg.is_file():
        raise FileNotFoundError(
            f"WM hydra.yaml not found: {hydra_cfg}\n"
            f"  Use --dino_ckpt_dir /workspace --dino_encoder wm_ckpt_18-27-17"
        )
    if not snapshot.is_file():
        raise FileNotFoundError(f"WM checkpoint not found: {snapshot}")
    print(f"[INFO] Loading WM from {ckpt_dir}")
    print(
        f"[INFO] visual_feature={args.visual_feature} "
        f"critic_fusion={args.critic_fusion}"
    )
    train_cfg = OmegaConf.load(str(hydra_cfg))
    wm = load_model(snapshot, train_cfg, train_cfg.num_action_repeat, device=args.device)
    for p in wm.parameters():
        p.requires_grad = True

    def make_env():
        return LatentHumanoidEnv(
            args,
            wm,
            args.device,
            args_cli,
            with_proprio=args.with_proprio,
            latent_h=args.latent_h,
            wandb_video_every=args.wandb_video_every,
        )

    if args.training_num != 1:
        print(f"[WARN] Forcing training_num=1 (was {args.training_num})")
        args.training_num = 1
    train_envs = DummyVectorEnv([make_env])

    state_space = train_envs.observation_space[0]
    action_space = train_envs.action_space[0]
    state_shape = state_space.shape
    action_shape = action_space.shape or action_space.n
    import gym

    policy_action_space = gym.spaces.Box(
        low=np.asarray(action_space.low, dtype=np.float32),
        high=np.asarray(action_space.high, dtype=np.float32),
        dtype=np.float32,
    )

    def _make_critic():
        critic = make_q_critic(
            args.critic_fusion,
            state_shape,
            action_shape,
            hidden_sizes=args.critic_net,
            activation=getattr(torch.nn, args.critic_activation),
            device=args.device,
        )
        optim = torch.optim.AdamW(
            critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay_pyhj
        )
        return critic, optim

    critic1, critic1_optim = _make_critic()
    critic2, critic2_optim = _make_critic()

    actor_net = Net(
        state_shape,
        hidden_sizes=args.control_net,
        activation=getattr(torch.nn, args.actor_activation),
        device=args.device,
    )
    # unbounded=True: mu in R, tanh applied inside SACPolicy.forward (standard SAC).
    actor1 = ActorProb(
        actor_net,
        action_shape,
        max_action=1.0,
        unbounded=True,
        device=args.device,
    ).to(args.device)
    actor1_optim = torch.optim.AdamW(actor1.parameters(), lr=args.actor_lr)

    if use_auto_alpha:
        target_entropy = float(-np.prod(action_shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=float(args.alpha_lr))
        alpha_arg = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha_arg = float(args.alpha)

    policy = avoid_SACPolicy_annealing(
        critic1=critic1,
        critic1_optim=critic1_optim,
        critic2=critic2,
        critic2_optim=critic2_optim,
        tau=args.tau,
        gamma=args.gamma_pyhj,
        alpha=alpha_arg,
        exploration_noise=None,
        deterministic_eval=True,
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=policy_action_space,
        actor1=actor1,
        actor1_optim=actor1_optim,
    )
    # Env HJ overlay expects policy.critic(z, a) — use twin-Q min via critic1 alias
    # for cheap logging; gate at test uses min(Q1,Q2).
    policy.critic = policy.critic1
    policy.warmup = False
    policy._attach_act_nom = False
    action_reg_coef = float(args.action_reg_coef)
    boundary_reg_coef = float(args.boundary_reg_coef)
    freeze_yaw = bool(getattr(args, "freeze_yaw", False))
    switch_collect = bool(getattr(args, "switch_collect", False))
    switch_threshold = float(getattr(args, "switch_threshold", 0.0))
    if switch_collect and boundary_reg_coef != 0.0:
        print(
            "[INFO] --switch_collect: dropping boundary="
            f"{boundary_reg_coef}; keeping λ_nom={action_reg_coef} "
            f"(MSE(a_sf, a_nom) if Q>=thr else MSE(a_sf, a_good))."
        )
        boundary_reg_coef = 0.0

    def _zero_yaw_act(act):
        """Force yaw (dim 2) to 0 in policy-space actions (tensor or ndarray)."""
        if isinstance(act, torch.Tensor):
            if act.shape[-1] < 3:
                return act
            out = act.clone()
            out[..., 2] = 0.0
            return out
        out = np.array(act, dtype=np.float32, copy=True)
        if out.ndim >= 1 and out.shape[-1] >= 3:
            out[..., 2] = 0.0
        return out

    def _waypoint_acts_policy(*, zero_yaw: bool = False) -> np.ndarray:
        fns = train_envs.get_env_attr("compute_waypoint_nav_action")
        acts_env = np.stack(
            [np.asarray(fn(), dtype=np.float32).reshape(-1) for fn in fns],
            axis=0,
        )
        acts = np.asarray(policy.map_action_inverse(acts_env), dtype=np.float32)
        if zero_yaw:
            acts = _zero_yaw_act(acts)
        return acts

    def _safe_side_acts_policy(*, zero_yaw: bool = False) -> np.ndarray:
        """Policy-space a_good toward the moving perp-sidestep of the nearest bin."""
        fns = train_envs.get_env_attr("compute_safe_side_nav_action")
        acts_env = np.stack(
            [np.asarray(fn(), dtype=np.float32).reshape(-1) for fn in fns],
            axis=0,
        )
        acts = np.asarray(policy.map_action_inverse(acts_env), dtype=np.float32)
        if zero_yaw:
            acts = _zero_yaw_act(acts)
        return acts

    def _batch_act_nom_tensor(batch, act_ref: torch.Tensor) -> torch.Tensor:
        act_nom = None
        pol = getattr(batch, "policy", None)
        if pol is not None and hasattr(pol, "act_nom"):
            act_nom = pol.act_nom
        if act_nom is None:
            act_nom = batch.act
        act_nom_t = torch.as_tensor(
            act_nom, dtype=act_ref.dtype, device=act_ref.device
        )
        if act_nom_t.ndim == 1:
            act_nom_t = act_nom_t.unsqueeze(0)
        # Train/BC: optional yaw=0 so we never imitate buffer yaw.
        if freeze_yaw:
            act_nom_t = _zero_yaw_act(act_nom_t)
        return act_nom_t

    def _batch_act_good_tensor(batch, act_ref: torch.Tensor) -> torch.Tensor | None:
        """Stored a_good (safe-side waypoint). None if the batch has no field."""
        pol = getattr(batch, "policy", None)
        act_good = None if pol is None else getattr(pol, "act_good", None)
        if act_good is None:
            return None
        act_good_t = torch.as_tensor(
            act_good, dtype=act_ref.dtype, device=act_ref.device
        )
        if act_good_t.ndim == 1:
            act_good_t = act_good_t.unsqueeze(0)
        if freeze_yaw:
            act_good_t = _zero_yaw_act(act_good_t)
        return act_good_t

    def _batch_use_sf_weight(batch, act_ref: torch.Tensor) -> torch.Tensor:
        pol = getattr(batch, "policy", None)
        n = int(act_ref.shape[0])
        use_sf = None
        if pol is not None:
            use_sf = getattr(pol, "use_good_reg", None)
            if use_sf is None:
                use_sf = getattr(pol, "use_sf", None)
        if use_sf is None:
            return act_ref.new_zeros(n)
        t = torch.as_tensor(use_sf, dtype=act_ref.dtype, device=act_ref.device).reshape(
            -1
        )
        if t.numel() == 1 and n > 1:
            t = t.expand(n)
        elif t.numel() != n:
            out = act_ref.new_zeros(n)
            m = min(n, int(t.numel()))
            out[:m] = t[:m]
            return out
        return t

    def _obs_to_tensor(obs) -> torch.Tensor:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=args.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        return obs_t

    def _q_nom_min(obs, act_nom) -> np.ndarray:
        """min(Q1, Q2)(z, a_nom) in policy space. Shape (B,)."""
        obs_t = _obs_to_tensor(obs)
        act_t = torch.as_tensor(act_nom, dtype=torch.float32, device=args.device)
        if act_t.ndim == 1:
            act_t = act_t.unsqueeze(0)
        with torch.no_grad():
            q = torch.min(
                policy.critic1(obs_t, act_t).flatten(),
                policy.critic2(obs_t, act_t).flatten(),
            )
        return q.detach().cpu().numpy().reshape(-1)

    def _mix_switch_act(act_sf, act_nom, use_sf: np.ndarray):
        """Keep raw SF act where use_sf else a_nom. Do not noise a_sf."""
        use_sf = np.asarray(use_sf, dtype=bool).reshape(-1)
        if isinstance(act_sf, torch.Tensor):
            act_nom_t = torch.as_tensor(
                act_nom, dtype=act_sf.dtype, device=act_sf.device
            )
            if act_nom_t.ndim == 1:
                act_nom_t = act_nom_t.unsqueeze(0)
            gate = torch.as_tensor(use_sf, device=act_sf.device).reshape(-1, 1)
            return torch.where(gate, act_sf, act_nom_t)
        act_sf_np = np.asarray(act_sf, dtype=np.float32)
        act_nom_np = np.asarray(act_nom, dtype=np.float32)
        if act_sf_np.ndim == 1:
            act_sf_np = act_sf_np.reshape(1, -1)
        if act_nom_np.ndim == 1:
            act_nom_np = act_nom_np.reshape(1, -1)
        out = np.array(act_nom_np, dtype=np.float32, copy=True)
        out[use_sf] = act_sf_np[use_sf]
        return out

    _orig_policy_forward = policy.forward

    def _note_use_sf(flags) -> None:
        fns = train_envs.get_env_attr("note_collect_use_sf")
        flags_np = np.asarray(flags, dtype=bool).reshape(-1)
        for i, fn in enumerate(fns):
            fn(bool(flags_np[i] if i < flags_np.size else False))

    def _frozen_act_good(n: int, hj_neg: np.ndarray, *, zero_yaw: bool) -> np.ndarray:
        """Policy-space a_good: compute on first HJ<0, freeze until HJ>=0."""
        out = np.zeros((n, 3), dtype=np.float32)
        hj_neg = np.asarray(hj_neg, dtype=bool).reshape(-1)
        if not np.any(hj_neg):
            return out
        held_list = train_envs.get_env_attr("held_a_good")
        held0 = held_list[0] if held_list else None
        if held0 is None:
            acts = _safe_side_acts_policy(zero_yaw=zero_yaw)
            held0 = np.asarray(acts, dtype=np.float32).reshape(-1, 3)[0]
            train_envs.set_env_attr("held_a_good", held0.copy())
        else:
            held0 = np.asarray(held0, dtype=np.float32).reshape(3)
        out[:] = held0.reshape(1, 3)
        return out

    def _attach_good_reg(pol_extra, obs, act_nom, n: int) -> None:
        """If λ>0, regularize on HJ(Q(a_nom))<0 regardless of collect controller."""
        if action_reg_coef <= 0.0:
            _note_use_sf(False)
            return
        q_nom = getattr(pol_extra, "q_nom", None)
        q_arr = None if q_nom is None else np.asarray(q_nom, dtype=np.float32).reshape(-1)
        if q_arr is None or q_arr.size == 0 or not np.isfinite(q_arr).all():
            q_arr = np.asarray(_q_nom_min(obs, act_nom), dtype=np.float32).reshape(-1)
            pol_extra.q_nom = q_arr
        hj_neg = q_arr < float(switch_threshold)
        hj_neg = np.asarray(hj_neg, dtype=np.bool_).reshape(-1)
        pol_extra.use_good_reg = hj_neg
        _note_use_sf(hj_neg)
        pol_extra.act_good = _frozen_act_good(n, hj_neg, zero_yaw=freeze_yaw)

    def _env_collect_controller() -> str:
        try:
            modes = train_envs.get_env_attr("collect_controller")
            return str(modes[0])
        except Exception:  # noqa: BLE001
            return "sf"

    def _as_act_like(act_ref, act_np: np.ndarray):
        if isinstance(act_ref, torch.Tensor):
            out = torch.as_tensor(act_np, dtype=act_ref.dtype, device=act_ref.device)
            if out.ndim == 1:
                out = out.unsqueeze(0)
            return out
        out = np.asarray(act_np, dtype=np.float32)
        if out.ndim == 1:
            out = out.reshape(1, -1)
        return out

    def _sac_forward(batch, state=None, input="obs", **kwargs):
        """SAC forward; collect may execute waypoint or Q-gate switch."""
        out = _orig_policy_forward(batch, state=state, input=input, **kwargs)
        if freeze_yaw:
            out.act = _zero_yaw_act(out.act)
        if getattr(policy, "_attach_act_nom", False):
            act_nom = _waypoint_acts_policy(zero_yaw=freeze_yaw)
            n = int(np.asarray(act_nom).reshape(-1, 3).shape[0])
            zeros_good = np.zeros((n, 3), dtype=np.float32)
            use_sf_np = np.zeros((n,), dtype=np.bool_)
            pol_extra = Batch(
                act_nom=act_nom,
                use_sf=use_sf_np,
                use_good_reg=np.zeros((n,), dtype=np.bool_),
                act_good=zeros_good,
                q_nom=np.full((n,), np.nan, dtype=np.float32),
            )
            mode = _env_collect_controller()
            if mode == "waypoint":
                out.act = _as_act_like(out.act, act_nom)
                policy.last_switch_use_sf = 0.0
            elif mode == "switch":
                q_nom = _q_nom_min(batch[input], act_nom)
                use_sf = q_nom < switch_threshold
                use_sf_np = np.asarray(use_sf, dtype=np.bool_).reshape(-1)
                pol_extra.q_nom = np.asarray(q_nom, dtype=np.float32)
                pol_extra.use_sf = use_sf_np
                out.act = _mix_switch_act(out.act, act_nom, use_sf)
                policy.last_q_nom = float(q_nom.reshape(-1)[0])
                policy.last_switch_use_sf = float(bool(use_sf_np.reshape(-1)[0]))
                ls = getattr(policy, "log_state", None)
                if isinstance(ls, dict):
                    ls["switch_n"] = int(ls.get("switch_n", 0)) + int(use_sf_np.size)
                    ls["switch_sf"] = int(ls.get("switch_sf", 0)) + int(use_sf_np.sum())
            else:
                # Fully SF collect: every step uses the actor.
                use_sf_np = np.ones((n,), dtype=np.bool_)
                pol_extra.use_sf = use_sf_np
                policy.last_switch_use_sf = 1.0
            _attach_good_reg(pol_extra, batch[input], act_nom, n)
            if np.isfinite(np.asarray(pol_extra.q_nom, dtype=np.float32)).any():
                q0 = float(np.asarray(pol_extra.q_nom, dtype=np.float32).reshape(-1)[0])
                policy.last_q_nom = q0
            out.policy = pol_extra
        return out

    policy.forward = _sac_forward  # type: ignore[method-assign]

    def learn_sac_humanoid(batch, **kwargs):
        """Twin-critic HJ update + SAC actor (optional freeze_yaw)."""
        del kwargs
        td1, critic1_loss = policy._mse_optimizer(
            batch, policy.critic1, policy.critic1_optim
        )
        td2, critic2_loss = policy._mse_optimizer(
            batch, policy.critic2, policy.critic2_optim
        )
        batch.weight = (td1 + td2) / 2.0

        actor_loss_v = 0.0
        nom_reg_v = 0.0
        good_reg_v = 0.0
        boundary_reg_v = 0.0
        nom_reg_safe_frac = 0.0
        alpha_loss_v = None
        act_abs_mean = 0.0
        act_sat_frac = 0.0

        if not policy.warmup:
            # Learn always uses the SF actor (not the switched env action).
            obs_result = policy(batch)
            act = obs_result.act  # yaw already 0 if --freeze_yaw
            q = torch.min(
                policy.critic1(batch.obs, act).flatten(),
                policy.critic2(batch.obs, act).flatten(),
            )
            actor_loss = (policy._alpha * obs_result.log_prob.flatten() - q).mean()
            nom_reg = act.new_zeros(())
            good_reg = act.new_zeros(())
            boundary_reg = act.new_zeros(())
            if action_reg_coef > 0.0 or boundary_reg_coef > 0.0:
                if action_reg_coef > 0.0:
                    # HJ(Q(a_nom))<0, any collect controller: pull π toward frozen a_good.
                    use_sf_w = _batch_use_sf_weight(batch, act)
                    act_good = _batch_act_good_tensor(batch, act)
                    if act_good is not None:
                        sq_good = (act - act_good).pow(2).mean(dim=-1)
                        valid = use_sf_w * (act_good.detach().abs().sum(dim=-1) > 1e-5)
                        denom = valid.sum().clamp(min=1.0)
                        good_reg = (sq_good * valid).sum() / denom
                        if float(valid.sum().item()) > 0.0:
                            actor_loss = actor_loss + action_reg_coef * good_reg
                        nom_reg_safe_frac = float((1.0 - use_sf_w).mean().item())
                if boundary_reg_coef > 0.0:
                    boundary_reg = torch.relu(act.abs() - 0.8).pow(2).mean()
                    actor_loss = actor_loss + boundary_reg_coef * boundary_reg

            policy.actor1_optim.zero_grad()
            actor_loss.backward()
            policy.actor1_optim.step()

            if policy._is_auto_alpha:
                log_prob = obs_result.log_prob.detach() + policy._target_entropy
                alpha_loss = -(policy._log_alpha * log_prob).mean()
                policy._alpha_optim.zero_grad()
                alpha_loss.backward()
                policy._alpha_optim.step()
                policy._alpha = policy._log_alpha.detach().exp()
                alpha_loss_v = float(alpha_loss.item())

            actor_loss_v = float(actor_loss.item())
            nom_reg_v = float(nom_reg.item())
            good_reg_v = float(good_reg.item())
            boundary_reg_v = float(boundary_reg.item())
            act_abs_mean = float(act.detach().abs().mean().item())
            act_sat_frac = float((act.detach().abs() > 0.95).float().mean().item())

        policy.sync_weight()
        result = {
            "loss/actor": actor_loss_v,
            "loss/critic1": float(critic1_loss.item()),
            "loss/critic2": float(critic2_loss.item()),
            "loss/critic": float(0.5 * (critic1_loss.item() + critic2_loss.item())),
            "loss/nom_reg": nom_reg_v,
            "loss/boundary_reg": boundary_reg_v,
            "train/actor_abs_mean": act_abs_mean,
            "train/actor_sat_frac": act_sat_frac,
        }
        if action_reg_coef > 0.0:
            result["train/nom_reg_safe_frac"] = nom_reg_safe_frac
            result["loss/good_reg"] = good_reg_v
        if alpha_loss_v is not None:
            result["loss/alpha"] = alpha_loss_v
            result["train/alpha"] = float(policy._alpha.item())
        elif not policy._is_auto_alpha:
            result["train/alpha"] = float(policy._alpha)
        return result

    policy.learn = learn_sac_humanoid  # type: ignore[method-assign]
    print(
        f"[INFO] SAC avoid SF (path pipeline): gamma={args.gamma_pyhj}, "
        f"auto_alpha={use_auto_alpha}, alpha={args.alpha}, "
        f"critic_warmup={args.critic_warmup_updates}, "
        f"λ_good={action_reg_coef} (any controller, HJ(Q(a_nom))<0; "
        f"freeze a_good until HJ>=0, hold_steps="
        f"{int(getattr(args, 'a_good_hold_steps', 0))}), "
        f"boundary={boundary_reg_coef}, "
        f"h_s=d_min-{float(args.lidar_distance_threshold):g}"
        + (
            f"+contact(ignore feet, cap={float(args.contact_hs):g})"
            if bool(args.include_contact_in_hs)
            else " (lidar only)"
        )
        + f", lidar_cone=±{float(args.lidar_h_half_fov_deg):g}°"
        + f", z_fuse_frames={int(getattr(args, 'z_fuse_frames', 5))}"
        + ", "
        f"vy=[{float(ENV_VY_RANGE[0]):g},{float(ENV_VY_RANGE[1]):g}], "
        f"yaw_limit={float(args.yaw_limit):g}, "
        f"freeze_yaw={freeze_yaw}, "
        f"alternate_collect={bool(args.alternate_collect)} "
        f"(waypoint ↔ switch), "
        + (
            f"nom_|v|~U[{float(args.nom_speed_min):g},{float(args.nom_speed_max):g}]/ep, "
            if bool(getattr(args, "randomize_nom_speed", False))
            else "nom_|v|=0.5, "
        )
        + _nom_act_noise_banner(args, sep=", ")
        + f"obstacles: absent_p={float(args.obstacle_absent_prob):g} "
        f"P(2|present)={float(args.two_obstacle_prob):g} max={int(args.max_n_obstacles)}"
    )

    if int(getattr(args, "wandb_video_every", 1)) != 0:
        args.wandb_video_every = 1
    log_state = {"env_step": 0, "update": 0, "switch_n": 0, "switch_sf": 0}
    train_envs.set_env_attr("wandb_video_every", int(args.wandb_video_every))
    train_envs.set_env_attr("log_state", log_state)
    train_envs.set_env_attr("policy_for_log", policy)
    policy.log_state = log_state
    policy.last_clean_act_env = None
    policy.last_q_nom = None
    policy.last_switch_use_sf = None

    resume_epoch = 0
    if args.resume_policy:
        load_policy_checkpoint(policy, args.resume_policy, args.device)
        resume_epoch = _parse_resume_epoch_from_path(args.resume_policy)
        if resume_epoch >= args.total_episodes:
            raise ValueError(
                f"resume epoch {resume_epoch} already >= total-episodes "
                f"{args.total_episodes}; nothing to train."
            )
        print(
            f"[INFO] Resumed at epoch {resume_epoch}; "
            f"continuing through epoch {args.total_episodes}"
        )

    start_epoch = resume_epoch + 1
    end_epoch = args.total_episodes

    _orig_log_update = logger.log_update_data

    def log_update_data_synced(update_result, step):
        log_state["update"] = int(step)
        env_step = int(log_state.get("env_step", 0))
        payload = {
            "trainer/env_step": float(env_step),
            "trainer/update": float(step),
            "loss/actor": float(update_result.get("loss/actor", 0.0)),
            "loss/critic": float(update_result.get("loss/critic", 0.0)),
            "loss/critic1": float(update_result.get("loss/critic1", 0.0)),
            "loss/critic2": float(update_result.get("loss/critic2", 0.0)),
        }
        for key in (
            "loss/nom_reg",
            "loss/boundary_reg",
            "loss/good_reg",
            "loss/alpha",
            "train/alpha",
            "train/actor_abs_mean",
            "train/actor_sat_frac",
            "train/nom_reg_safe_frac",
        ):
            if key in update_result:
                payload[key] = float(update_result[key])
        wandb.log(payload)
        return _orig_log_update(update_result, step)

    logger.log_update_data = log_update_data_synced  # type: ignore[method-assign]

    def exploration_noise_stash(act, batch):
        """Do not add collect noise. a_nom noise is in waypoint |v| only; a_sf stays raw."""
        del batch
        try:
            if freeze_yaw:
                act = _zero_yaw_act(act)
            if isinstance(act, torch.Tensor):
                act_np = act.detach().cpu().numpy()
            else:
                act_np = np.asarray(act)
            act_env = policy.map_action(np.array(act_np, dtype=np.float64, copy=True))
            policy.last_clean_act_env = np.asarray(act_env, dtype=np.float64).reshape(
                -1, 3
            )[0]
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] act stash failed: {exc}")
            policy.last_clean_act_env = None
        return act

    policy.exploration_noise = exploration_noise_stash  # type: ignore[method-assign]

    def train_fn(epoch: int, step_idx: int):
        del step_idx
        alpha_v = (
            float(policy._alpha.item())
            if torch.is_tensor(policy._alpha)
            else float(policy._alpha)
        )
        payload = {
            "trainer/env_step": float(log_state.get("env_step", 0)),
            "train/epoch_alpha": alpha_v,
            "train/epoch": float(epoch),
        }
        n_sw = int(log_state.get("switch_n", 0))
        if switch_collect and n_sw > 0:
            payload["switch/sf_frac_cum"] = float(log_state.get("switch_sf", 0)) / n_sw
        wandb.log(payload)

    buffer = VectorReplayBuffer(args.buffer_size, args.training_num)
    if bool(getattr(args, "skip_arena_oob_from_buffer", True)):
        buffer = SkipUpdateReplayBuffer(buffer)
        print("[INFO] Arena-edge resets are dropped from the replay buffer (no update).")
    # exploration_noise=True only calls our stash (no Gaussian); SAC samples in forward.
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    _orig_collect = train_collector.collect

    def _collect_with_act_nom(*a, **kw):
        policy._attach_act_nom = True
        try:
            return _orig_collect(*a, **kw)
        finally:
            policy._attach_act_nom = False

    train_collector.collect = _collect_with_act_nom  # type: ignore[method-assign]

    # Scene: start→goal (or start + 2 perp vias); bins on the SG perpendicular.
    def _apply_path_scene_layout() -> None:
        train_envs.set_env_attr("obstacle_absent_prob", float(args.obstacle_absent_prob))
        train_envs.set_env_attr("two_obstacle_prob", float(args.two_obstacle_prob))
        train_envs.set_env_attr("path_obstacle_layout", True)
        train_envs.set_env_attr("filter_lidar_to_active_obstacles", True)
        train_envs.set_env_attr("randomize_obstacle", False)
        train_envs.set_env_attr("advance_passed_terminal", False)
        train_envs.set_env_attr("spawn_hemisphere_pass", False)
        train_envs.set_env_attr("force_pass_side", None)
        train_envs.set_env_attr("include_middle_pass", False)
        train_envs.set_env_attr("waypoint_layout", str(args.waypoint_layout))
        train_envs.set_env_attr("arena_x_min", float(args.arena_x_min))
        train_envs.set_env_attr("arena_x_max", float(args.arena_x_max))
        train_envs.set_env_attr("arena_y_min", float(args.arena_y_min))
        train_envs.set_env_attr("arena_y_max", float(args.arena_y_max))
        train_envs.set_env_attr("use_arena_bounds", True)
        train_envs.set_env_attr("perp_offset", float(args.perp_offset))
        train_envs.set_env_attr(
            "perp_keepout", float(getattr(args, "perp_keepout", 0.25))
        )
        train_envs.set_env_attr("min_start_goal_dist", float(args.min_start_goal_dist))
        train_envs.set_env_attr(
            "lidar_distance_threshold", float(args.lidar_distance_threshold)
        )
        train_envs.set_env_attr(
            "lidar_h_half_fov_deg", float(args.lidar_h_half_fov_deg)
        )
        train_envs.set_env_attr(
            "include_contact_in_hs", bool(args.include_contact_in_hs)
        )
        train_envs.set_env_attr("contact_hs", float(args.contact_hs))
        train_envs.set_env_attr(
            "randomize_nom_speed", bool(getattr(args, "randomize_nom_speed", False))
        )
        train_envs.set_env_attr(
            "nom_speed_min", float(getattr(args, "nom_speed_min", 0.05))
        )
        train_envs.set_env_attr(
            "nom_speed_max", float(getattr(args, "nom_speed_max", 0.95))
        )
        train_envs.set_env_attr(
            "nom_speed_mean", float(getattr(args, "nom_speed_mean", 0.5))
        )
        train_envs.set_env_attr(
            "nom_speed_std", float(getattr(args, "nom_speed_std", 0.3))
        )
        train_envs.set_env_attr(
            "nom_act_noise", bool(getattr(args, "nom_act_noise", False))
        )
        train_envs.set_env_attr(
            "nom_act_noise_std", float(getattr(args, "nom_act_noise_std", 0.2))
        )
        train_envs.set_env_attr(
            "nom_act_noise_stds", str(getattr(args, "nom_act_noise_stds", "") or "")
        )
        train_envs.set_env_attr(
            "nom_act_noise_dist",
            str(getattr(args, "nom_act_noise_dist", "uniform")),
        )
        train_envs.set_env_attr(
            "sidestep_hold_steps", int(getattr(args, "a_good_hold_steps", 0))
        )
        train_envs.set_env_attr(
            "skip_arena_oob_from_buffer",
            bool(args.skip_arena_oob_from_buffer),
        )
        train_envs.set_env_attr(
            "alternate_collect_controllers", bool(args.alternate_collect)
        )
        train_envs.set_env_attr(
            "z_fuse_frames", int(getattr(args, "z_fuse_frames", 5))
        )

    arena_desc = (
        f"x[{float(args.arena_x_min):g},{float(args.arena_x_max):g}] "
        f"y[{float(args.arena_y_min):g},{float(args.arena_y_max):g}]"
    )
    if str(args.waypoint_layout) == "start_goal":
        path_desc = (
            "start → goal (straight); bins |offset|~U["
            f"{float(getattr(args, 'perp_keepout', 0.25)):g},"
            f"{float(args.perp_offset):g}] m, side L/L/R/R every 2 eps; "
            f"min |G-S|={float(args.min_start_goal_dist):g} m"
        )
    else:
        path_desc = (
            "start → 2 vias in a ±"
            f"{float(args.perp_offset):g} m band → goal; "
            f"min |G-S|={float(args.min_start_goal_dist):g} m"
        )

    _actor_forward = policy.forward
    _expl_fn = policy.exploration_noise

    _apply_path_scene_layout()
    print(
        "[INFO] Collecting initial transitions with WaypointNavController "
        f"({path_desc}; arena {arena_desc}; "
        f"h_s=d_min-{float(args.lidar_distance_threshold):g}"
        f"{'+contact' if bool(args.include_contact_in_hs) else ' lidar-only'}; "
        + (
            f"nom_|v|~U[{float(args.nom_speed_min):g},{float(args.nom_speed_max):g}]/ep; "
            if bool(getattr(args, "randomize_nom_speed", False))
            else "nom_|v|=0.5; "
        )
        + _nom_act_noise_banner(args, sep="; ")
        + f"lidar_cone=±{float(args.lidar_h_half_fov_deg):g}°; "
        f"vy=[{float(ENV_VY_RANGE[0]):g},{float(ENV_VY_RANGE[1]):g}]; "
        f"yaw=±{float(args.yaw_limit):g}; "
        f"obstacles: absent_p={float(args.obstacle_absent_prob):g}, "
        f"P(2|present)={float(args.two_obstacle_prob):g}, "
        f"perp U[{float(getattr(args, 'perp_keepout', 0.25)):g},"
        f"{float(args.perp_offset):g}] L/L/R/R"
        f"{'; resume' if args.resume_policy else ''})..."
    )

    def _waypoint_forward(batch, state=None, **kwargs):
        del batch, state, kwargs
        acts = _waypoint_acts_policy()
        n = int(np.asarray(acts).reshape(-1, 3).shape[0])
        pol_extra = Batch(
            act_nom=acts.copy(),
            use_sf=np.zeros((n,), dtype=np.bool_),
            use_good_reg=np.zeros((n,), dtype=np.bool_),
            act_good=np.zeros((n, 3), dtype=np.float32),
            q_nom=np.full((n,), np.nan, dtype=np.float32),
        )
        return Batch(act=acts, state=None, policy=pol_extra)

    def _waypoint_expl(act, batch):
        del batch
        try:
            if isinstance(act, torch.Tensor):
                act_np = act.detach().cpu().numpy()
            else:
                act_np = np.asarray(act)
            act_env = policy.map_action(np.array(act_np, dtype=np.float64, copy=True))
            policy.last_clean_act_env = np.asarray(act_env, dtype=np.float64).reshape(
                -1, 3
            )[0]
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] waypoint act stash failed: {exc}")
            policy.last_clean_act_env = None
        return act

    policy.forward = _waypoint_forward  # type: ignore[method-assign]
    policy.exploration_noise = _waypoint_expl  # type: ignore[method-assign]
    warmup_n_step = max(0, int(getattr(args, "buffer_warmup_steps", 4000)))
    warmup_chunk = 50
    print(
        f"[INFO] Warm-up collect n_step={warmup_n_step} "
        f"(progress every {warmup_chunk} steps).",
        flush=True,
    )
    collected = 0
    ep_total = 0
    t0 = time.time()
    while collected < warmup_n_step:
        n = min(warmup_chunk, warmup_n_step - collected)
        stats = train_collector.collect(n)
        collected += int(stats.get("n/st", n))
        ep_total += int(stats.get("n/ep", 0))
        elapsed = time.time() - t0
        sps = collected / max(elapsed, 1e-6)
        print(
            f"[INFO] buffer warm-up: {collected}/{warmup_n_step} steps "
            f"({100.0 * collected / warmup_n_step:.0f}%), "
            f"episodes={ep_total}, {sps:.2f} steps/s, elapsed={elapsed:.0f}s",
            flush=True,
        )
    print(
        f"[INFO] Warm-up collect done: {collected} steps, {ep_total} episodes, "
        f"{time.time() - t0:.0f}s total.",
        flush=True,
    )
    policy.forward = _actor_forward  # type: ignore[method-assign]
    policy.exploration_noise = _expl_fn  # type: ignore[method-assign]
    _apply_path_scene_layout()
    if bool(args.alternate_collect):
        train_envs.set_env_attr("_collect_ep_toggle", 0)
        train_envs.set_env_attr("collect_controller", "waypoint")
        print(
            "[INFO] Formal collect alternates per episode: "
            "waypoint-only ↔ Q-gate switch (a_nom if Q>=0 else SF)."
        )
    print(
        "[INFO] Buffer done; restoring SAC actor "
        f"(learn always SF; freeze_yaw={freeze_yaw}; {path_desc}; "
        f"arena {arena_desc}, OOB reset skip_update="
        f"{bool(args.skip_arena_oob_from_buffer)}). "
        f"Obstacles: absent_p={float(args.obstacle_absent_prob):g}, "
        f"P(2|present)={float(args.two_obstacle_prob):g}."
    )

    n_critic_wu = int(getattr(args, "critic_warmup_updates", 4000))
    if n_critic_wu > 0 and not args.resume_policy:
        print(f"[INFO] Critic-only warmup: {n_critic_wu} updates...")
        policy.warmup = True
        for i in range(1, n_critic_wu + 1):
            metrics = policy.update(args.batch_size_pyhj, buffer)
            log_state["update"] = int(log_state.get("update", 0)) + 1
            wandb.log(
                {
                    "trainer/env_step": float(log_state.get("env_step", 0)),
                    "trainer/update": float(log_state["update"]),
                    "loss/critic": float(metrics["loss/critic"]),
                    "loss/critic1": float(metrics["loss/critic1"]),
                    "loss/critic2": float(metrics["loss/critic2"]),
                    "train/critic_warmup": 1.0,
                }
            )
            if i == 1 or i % 100 == 0 or i == n_critic_wu:
                print(
                    f"  critic_warmup {i}/{n_critic_wu}: "
                    f"loss/critic={metrics['loss/critic']:.4f}"
                )
        policy.warmup = False
        print("[INFO] Critic-only warmup done.")
    else:
        policy.warmup = False

    n_actor_bc = int(getattr(args, "actor_bc_warmup_updates", 0))
    if n_actor_bc > 0 and not args.resume_policy:
        print(f"[INFO] Actor BC warmup: {n_actor_bc} updates (imitate waypoint)...")
        was_training = policy.training
        policy.eval()  # deterministic mean for BC
        for i in range(1, n_actor_bc + 1):
            batch, _ = buffer.sample(args.batch_size_pyhj)
            act_pred = policy(batch).act
            act_demo = _batch_act_nom_tensor(batch, act_pred)
            bc_loss = torch.nn.functional.mse_loss(act_pred, act_demo)
            policy.actor1_optim.zero_grad()
            bc_loss.backward()
            policy.actor1_optim.step()
            log_state["update"] = int(log_state.get("update", 0)) + 1
            wandb.log(
                {
                    "trainer/env_step": float(log_state.get("env_step", 0)),
                    "trainer/update": float(log_state["update"]),
                    "loss/actor_bc": float(bc_loss.item()),
                    "train/actor_bc_warmup": 1.0,
                }
            )
            if i == 1 or i % 100 == 0 or i == n_actor_bc:
                print(
                    f"  actor_bc_warmup {i}/{n_actor_bc}: "
                    f"loss/actor_bc={bc_loss.item():.4f}"
                )
        policy.train(was_training)
        print("[INFO] Actor BC warmup done.")

    log_path = Path(f"runs/sac_hj_humanoid_path/{run_name}")
    print(f"[INFO] Training epochs {start_epoch}..{end_epoch}; ckpts -> {log_path}")
    switch_traj_dir = log_path / "switch_traj"
    switch_traj_dir.mkdir(parents=True, exist_ok=True)
    train_envs.set_env_attr("switch_traj_dir", str(switch_traj_dir))
    train_envs.set_env_attr(
        "failure_set_radius", float(args.lidar_distance_threshold)
    )
    train_envs.set_env_attr("record_switch_traj", True)
    print(
        "[INFO] Formal-train switch episodes will save 2D traj PNGs -> "
        f"{switch_traj_dir} (obstacles as blue dots, dashed r="
        f"{float(args.lidar_distance_threshold):g} failure set)."
    )
    for epoch in range(start_epoch, end_epoch + 1):
        print(f"\n=== Epoch {epoch}/{end_epoch} ===")
        stats = offpolicy_trainer(
            policy=policy,
            train_collector=train_collector,
            test_collector=None,
            max_epoch=1,
            step_per_epoch=args.step_per_epoch,
            step_per_collect=args.step_per_collect,
            episode_per_test=args.test_num,
            batch_size=args.batch_size_pyhj,
            update_per_step=args.update_per_step,
            stop_fn=lambda r: False,
            train_fn=train_fn,
            save_best_fn=None,
            logger=logger,
        )

        numeric = {
            "train/epoch": int(epoch),
            "trainer/env_step": float(log_state.get("env_step", 0)),
            "trainer/update": float(log_state.get("update", 0)),
        }
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                numeric[f"train/{k}"] = v
            elif isinstance(v, np.generic):
                numeric[f"train/{k}"] = float(v)
        wandb.log(numeric)

        save_epoch_checkpoint(policy, log_path, epoch, keep=2)

    print("[INFO] Training complete.")
    simulation_app.close()


if __name__ == "__main__":
    main()
