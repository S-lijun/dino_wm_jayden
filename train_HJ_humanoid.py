"""DDPG HJ safety-filter training on Isaac G1 latent space."""

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

parser = argparse.ArgumentParser("DDPG HJ on DINO latent Humanoid (Isaac G1)")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--dino_ckpt_dir",
    type=str,
    default="/storage1/sibai/Active/ihab/research_new/checkpt_dino/outputs2/cargoal",
    help="Where to find the DINO-WM checkpoints",
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
    default="dino",
    help="Encoder subfolder under dino_ckpt_dir",
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
    help=(
        "Path to a saved .../epoch_id_N/policy.pth. "
        "Loads weights and continues training from epoch N+1 up to --total-episodes "
        "(from YAML / CLI, e.g. 120)."
    ),
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
    help="Soft |y| corridor (meters). <=0 disables (default). >0 truncates when |y|>bound.",
)
parser.add_argument(
    "--critic_warmup_updates",
    type=int,
    default=1000,
    help=(
        "After buffer warm-up collect, run this many critic-only updates "
        "(policy.warmup=True, actor frozen) before joint training."
    ),
)
parser.add_argument(
    "--actor_bc_warmup_updates",
    type=int,
    default=0,
    help=(
        "Optional: after critic warm-up, behavior-clone the actor to buffer "
        "actions for this many updates before joint RL. Default 0 (off)."
    ),
)
parser.add_argument(
    "--critic_action_samples",
    type=int,
    default=64,
    help="Deprecated (critic-sample control removed). Kept for CLI compat; ignored.",
)
parser.add_argument(
    "--action_reg_coef",
    type=float,
    default=0.0,
    help=(
        "Weight λ_nom for MSE(actor, waypoint a_nom) in policy space [-1,1]. "
        "0 disables. Start around 0.1; pulls SF toward nominal vx/vy/yaw."
    ),
)
parser.add_argument(
    "--boundary_reg_coef",
    type=float,
    default=0.5,
    help=(
        "Extra penalty when |act| > 0.8 (near tanh corners ±1). "
        "Set 0 to disable."
    ),
)
# --device is already registered by AppLauncher.add_app_launcher_args()

args_cli, remaining = parser.parse_known_args()

sys.path.insert(0, os.path.join(ISAACLAB_ROOT, "scripts/demos"))
from visual_obs_utils import configure_app_for_visual, resolve_visual_mode

_visual_mode = resolve_visual_mode(args_cli)
configure_app_for_visual(args_cli, _visual_mode)
torch_device = getattr(args_cli, "device", "cuda:0")
if not bool(getattr(args_cli, "headless", False)):
    print(
        "[WARN] Running WITHOUT --headless. GUI + rtx_rgb will freeze as "
        "'Not Responding' on Windows. Keep rtx_rgb, but MUST add --headless:\n"
        "  python train_HJ_humanoid.py --headless --visual_mode rtx_rgb ..."
    )
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Prefer repo root after AppLauncher mutates sys.path (Isaac cv2/utils shadowing).
sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------
# Imports after AppLauncher
# ---------------------------------------------------------------------
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
from PyHJ.exploration import GaussianNoise
from PyHJ.utils import WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.policy import avoid_DDPGPolicy_annealing

# Progress bar: only actor/critic. Reg / sat metrics still go to wandb via logger.
_PROGRESS_BAR_LOSS_KEYS = ("loss/actor", "loss/critic")


def _log_update_data_bar_filter(self, data, losses):
    for k in losses.keys():
        self.stat[k].add(losses[k])
        losses[k] = self.stat[k].get()
        if k in _PROGRESS_BAR_LOSS_KEYS:
            data[k] = f"{losses[k]:.3f}"
    self.logger.log_update_data(losses, self.gradient_step)


OffpolicyTrainer.log_update_data = _log_update_data_bar_filter  # type: ignore[method-assign]

from wm_load import load_model  # not plan.py — avoids env.venv / utils collision
from env.isaac.latent_humanoid_env import LatentHumanoidEnv
from env.isaac.ckpt_utils import save_epoch_checkpoint


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


def _parse_resume_epoch_from_path(policy_path: str | Path) -> int:
    """Infer N from '.../epoch_id_N/policy.pth'."""
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
    """Load a previously saved policy.state_dict() into ``policy``."""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"--resume_policy not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(state, strict=True)
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

    os.environ.setdefault("MPLCONFIGDIR", "/storage1/sibai/Active/ihab/tmp")
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

    from datetime import datetime

    timestamp = datetime.now().strftime("%m%d_%H%M")
    wandb.init(
        project="ddpg-hj-latent-humanoid",
        name=f"ddpg-{args.dino_encoder}-{timestamp}",
        config=vars(args),
    )
    # All rollout + loss curves share env_step as x-axis.
    wandb.define_metric("trainer/env_step")
    wandb.define_metric("trainer/update")
    wandb.define_metric("safety/*", step_metric="trainer/env_step")
    wandb.define_metric("actor_action/*", step_metric="trainer/env_step")
    wandb.define_metric("rollout/*", step_metric="trainer/env_step")
    wandb.define_metric("loss/*", step_metric="trainer/env_step")
    wandb.define_metric("train/*", step_metric="trainer/env_step")
    writer = SummaryWriter(log_dir=f"runs/ddpg_hj_humanoid/{args.dino_encoder}-{timestamp}/logs")
    # update_interval=1: log every grad step (default 1000 would skip almost everything).
    wb_logger = WandbLogger(update_interval=1, train_interval=10**9, test_interval=10**9)
    wb_logger.load(writer)
    logger = wb_logger

    ckpt_dir = Path(args.dino_ckpt_dir)
    hydra_cfg = ckpt_dir / "hydra.yaml"
    snapshot = ckpt_dir / "checkpoints" / "model_latest.pth"
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

    # Isaac Sim allows only one SimulationContext per process. Do not create a
    # second DummyVectorEnv for "test" — offpolicy_trainer already uses
    # test_collector=None below.
    if args.training_num != 1:
        print(f"[WARN] Forcing training_num=1 (was {args.training_num})")
        args.training_num = 1
    train_envs = DummyVectorEnv([make_env])

    state_space = train_envs.observation_space[0]
    action_space = train_envs.action_space[0]
    state_shape = state_space.shape
    action_shape = action_space.shape or action_space.n
    # Actor outputs tanh in [-1, 1]; map_action linearly scales to env bounds
    # (vx [0,0.8], vy [-0.5,0.5], yaw [-0.5,0.5]). Do NOT use high as max_action —
    # that Dubins/CarGoal pattern assumes symmetric [-high, high] actions.
    max_action = 1.0
    # PyHJ map_action checks gym.spaces.Box; env uses gymnasium.Box.
    import gym

    policy_action_space = gym.spaces.Box(
        low=np.asarray(action_space.low, dtype=np.float32),
        high=np.asarray(action_space.high, dtype=np.float32),
        dtype=np.float32,
    )

    critic_net = Net(
        state_shape,
        action_shape,
        hidden_sizes=args.critic_net,
        activation=getattr(torch.nn, args.critic_activation),
        concat=True,
        device=args.device,
    )
    critic = Critic(critic_net, device=args.device).to(args.device)
    critic_optim = torch.optim.AdamW(
        critic.parameters(), lr=args.critic_lr, weight_decay=args.weight_decay_pyhj
    )

    actor_net = Net(
        state_shape,
        hidden_sizes=args.control_net,
        activation=getattr(torch.nn, args.actor_activation),
        device=args.device,
    )
    actor = Actor(
        actor_net,
        action_shape,
        max_action=max_action,
        device=args.device,
    ).to(args.device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)

    policy = avoid_DDPGPolicy_annealing(
        critic=critic,
        critic_optim=critic_optim,
        tau=args.tau,
        gamma=args.gamma_pyhj,
        exploration_noise=GaussianNoise(sigma=args.exploration_noise),
        reward_normalization=args.rew_norm,
        estimation_step=args.n_step,
        action_space=policy_action_space,
        actor=actor,
        actor_optim=actor_optim,
        actor_gradient_steps=args.actor_gradient_steps,
    )
    # PyHJ default new_expl=True replaces acts with random when Q(rand)>=0;
    # that biases the buffer and worsens actor collapse. Use standard Gaussian noise.
    policy.new_expl = False
    action_reg_coef = float(args.action_reg_coef)  # λ_nom: MSE to waypoint a_nom
    boundary_reg_coef = float(args.boundary_reg_coef)
    policy._attach_act_nom = False  # True only inside collector.collect

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
        """Current waypoint cmds in policy space, shape (n_env, 3)."""
        fns = train_envs.get_env_attr("compute_waypoint_nav_action")
        acts_env = np.stack(
            [np.asarray(fn(), dtype=np.float32).reshape(-1) for fn in fns],
            axis=0,
        )
        acts = np.asarray(policy.map_action_inverse(acts_env), dtype=np.float32)
        if zero_yaw:
            acts = _zero_yaw_act(acts)
        return acts

    def _batch_act_nom_tensor(batch, act_ref: torch.Tensor) -> torch.Tensor:
        """Nominal from buffer.policy.act_nom; fallback batch.act. Train: yaw→0."""
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
        # Joint-train / BC: never imitate yaw (buffer may still store controller yaw).
        return _zero_yaw_act(act_nom_t)

    _orig_policy_forward = policy.forward

    def _actor_forward_maybe_nom(batch, state=None, model="actor", input="obs", **kwargs):
        """Actor forward with yaw=0; during collect store a_nom (yaw=0) too."""
        out = _orig_policy_forward(
            batch, state=state, model=model, input=input, **kwargs
        )
        out.act = _zero_yaw_act(out.act)
        if getattr(policy, "_attach_act_nom", False):
            # Training nominal: same waypoint vx/vy, yaw forced to 0.
            act_nom = _waypoint_acts_policy(zero_yaw=True)
            out.policy = Batch(act_nom=act_nom)
        return out

    policy.forward = _actor_forward_maybe_nom  # type: ignore[method-assign]

    def learn_anti_tanh(batch, **kwargs):
        """Joint critic + actor: max Q + λ_nom||π - a_nom||² + boundary reg (yaw=0)."""
        del kwargs
        td, critic_loss = policy._mse_optimizer(
            batch, policy.critic, policy.critic_optim
        )
        batch.weight = td
        if not policy.warmup:
            for _ in range(policy.actor_gradient_steps):
                act = policy(batch, model="actor").act  # yaw already 0
                safety_loss = -policy.critic(batch.obs, act).mean()
                act_nom = _batch_act_nom_tensor(batch, act)  # yaw 0
                nom_reg = torch.nn.functional.mse_loss(act, act_nom)
                boundary_reg = torch.relu(act.abs() - 0.8).pow(2).mean()
                actor_loss = safety_loss + boundary_reg_coef * boundary_reg
                if action_reg_coef > 0.0:
                    actor_loss = actor_loss + action_reg_coef * nom_reg
                policy.actor_optim.zero_grad()
                actor_loss.backward()
                policy.actor_optim.step()
            act_abs_mean = float(act.detach().abs().mean().item())
            act_sat_frac = float((act.detach().abs() > 0.95).float().mean().item())
        else:
            actor_loss = torch.tensor(0.0)
            nom_reg = actor_loss
            boundary_reg = actor_loss
            act_abs_mean = 0.0
            act_sat_frac = 0.0
        policy.sync_weight()
        return {
            "loss/actor": float(actor_loss.item()),
            "loss/critic": float(critic_loss.item()),
            "loss/action_reg": float(nom_reg.item()) if not policy.warmup else 0.0,
            "loss/nom_reg": float(nom_reg.item()) if not policy.warmup else 0.0,
            "loss/boundary_reg": (
                float(boundary_reg.item()) if not policy.warmup else 0.0
            ),
            "train/actor_abs_mean": act_abs_mean,
            "train/actor_sat_frac": act_sat_frac,
        }

    policy.learn = learn_anti_tanh  # type: ignore[method-assign]
    print(
        f"[INFO] actor+critic SF: gamma={args.gamma_pyhj}, "
        f"actor_lr={args.actor_lr}, "
        f"critic_warmup_updates={args.critic_warmup_updates}, "
        f"actor_bc_warmup_updates={args.actor_bc_warmup_updates}, "
        f"exploration_noise_start={args.exploration_noise} "
        f"(anneal → 0.1 over training), "
        f"new_expl=False, "
        f"action_reg_coef(λ_nom)={action_reg_coef}, "
        f"boundary_reg_coef={boundary_reg_coef}, "
        f"train yaw frozen to 0 (SF + a_nom); buffer unchanged"
    )
    # Force every-episode video uploads (CLI 0 still disables).
    if int(getattr(args, "wandb_video_every", 1)) != 0:
        args.wandb_video_every = 1
    log_state = {"env_step": 0, "update": 0}
    train_envs.set_env_attr("wandb_video_every", int(args.wandb_video_every))
    train_envs.set_env_attr("log_state", log_state)
    # Enable per-frame HJ overlay + safety/l, safety/hj wandb scalars in the env.
    train_envs.set_env_attr("policy_for_log", policy)
    policy.log_state = log_state
    policy.last_clean_act_env = None

    # Warm-start from a previous safety-filter checkpoint (actor+critic+targets).
    # Continues until args.total_episodes (same target as a fresh run, e.g. 120).
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

    # Progress-bar losses are MovAvg-smoothed inside OffpolicyTrainer.log_update_data.
    # Log THAT same smoothed dict to wandb (not the raw per-learn instantaneous loss).
    _orig_log_update = logger.log_update_data

    def log_update_data_synced(update_result, step):
        # Smoothed losses (same MovAvg as the progress bar), x-axis = env_step.
        log_state["update"] = int(step)
        env_step = int(log_state.get("env_step", 0))
        payload = {
            "trainer/env_step": float(env_step),
            "trainer/update": float(step),
            "loss/actor": float(update_result["loss/actor"]),
            "loss/critic": float(update_result["loss/critic"]),
        }
        for key in (
            "loss/action_reg",
            "loss/nom_reg",
            "loss/boundary_reg",
            "train/actor_abs_mean",
            "train/actor_sat_frac",
        ):
            if key in update_result:
                payload[key] = float(update_result[key])
        wandb.log(payload)
        return _orig_log_update(update_result, step)

    logger.log_update_data = log_update_data_synced  # type: ignore[method-assign]

    # Stash deterministic SF action (pre-noise) for env-side logging; do not wandb.log here.
    orig_exploration_noise = policy.exploration_noise

    def exploration_noise_stash_clean(act, batch):
        act = _zero_yaw_act(act)
        try:
            if isinstance(act, torch.Tensor):
                act_np = act.detach().cpu().numpy()
            else:
                act_np = np.asarray(act)
            act_env = policy.map_action(np.array(act_np, dtype=np.float64, copy=True))
            row = np.asarray(act_env, dtype=np.float64).reshape(-1, 3)[0]
            row = np.asarray(row, dtype=np.float64).reshape(-1)
            if row.size >= 3:
                row[2] = 0.0
            policy.last_clean_act_env = row
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] noiseless actor stash failed: {exc}")
            policy.last_clean_act_env = None
        return _zero_yaw_act(orig_exploration_noise(act, batch))

    policy.exploration_noise = exploration_noise_stash_clean

    def train_fn(epoch: int, step_idx: int):
        # Linearly anneal exploration noise: start (e.g. 0.3) → 0.1 by last epoch.
        del step_idx
        start_sigma = float(args.exploration_noise)
        end_sigma = 0.1
        if end_epoch <= start_epoch:
            sigma = end_sigma
        else:
            frac = (epoch - start_epoch) / float(end_epoch - start_epoch)
            frac = min(1.0, max(0.0, frac))
            sigma = start_sigma + (end_sigma - start_sigma) * frac
        if getattr(policy, "_noise", None) is not None:
            policy._noise._sigma = float(sigma)
        wandb.log(
            {
                "trainer/env_step": float(log_state.get("env_step", 0)),
                "train/exploration_sigma": float(sigma),
            }
        )

    buffer = VectorReplayBuffer(args.buffer_size, args.training_num)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    # Only query waypoint a_nom during env collect (not during learn / target_q).
    _orig_collect = train_collector.collect

    def _collect_with_act_nom(*args, **kwargs):
        policy._attach_act_nom = True
        try:
            return _orig_collect(*args, **kwargs)
        finally:
            policy._attach_act_nom = False

    train_collector.collect = _collect_with_act_nom  # type: ignore[method-assign]

    # Replay buffer is not restored; collect a fresh warm-up even when resuming weights.
    # Initial buffer: region waypoint nav (DataCollection style), not random actor.
    print(
        "[INFO] Collecting initial transitions with WaypointNavController "
        "(front -> cycle left|right|middle as goal; bin FIXED; "
        "middle = collision demos; no behind-bin back)..."
    )
    train_envs.set_env_attr("randomize_obstacle", False)
    train_envs.set_env_attr("include_middle_pass", True)
    _actor_forward = policy.forward
    _expl_fn = policy.exploration_noise

    def _waypoint_forward(batch, state=None, **kwargs):
        del batch, state, kwargs
        # Buffer unchanged: full controller yaw kept in act / act_nom.
        acts = _waypoint_acts_policy(zero_yaw=False)
        return Batch(act=acts, state=None, policy=Batch(act_nom=acts.copy()))

    def _waypoint_expl(act, batch):
        # Stash env cmd for actor_action/* logs; do not add Gaussian noise.
        try:
            if isinstance(act, torch.Tensor):
                act_np = act.detach().cpu().numpy()
            else:
                act_np = np.asarray(act)
            act_env = policy.map_action(np.array(act_np, dtype=np.float64, copy=True))
            row = np.asarray(act_env, dtype=np.float64).reshape(-1, 3)[0]
            policy.last_clean_act_env = row
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] waypoint act stash failed: {exc}")
            policy.last_clean_act_env = None
        return act

    policy.forward = _waypoint_forward  # type: ignore[method-assign]
    policy.exploration_noise = _waypoint_expl  # type: ignore[method-assign]
    warmup_n_step = 1000
    warmup_chunk = 50  # print progress every N env steps (Isaac RTX is slow)
    print(
        f"[INFO] Warm-up collect n_step={warmup_n_step} "
        f"(progress every {warmup_chunk} steps); "
        f"wandb video every {int(args.wandb_video_every)} episode(s); "
        f"HJ overlay=Q(z, a_controller); store policy.act_nom.",
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
            f"episodes={ep_total}, "
            f"{sps:.2f} steps/s, elapsed={elapsed:.0f}s",
            flush=True,
        )
    print(
        f"[INFO] Warm-up collect done: {collected} steps, {ep_total} episodes, "
        f"{time.time() - t0:.0f}s total.",
        flush=True,
    )
    policy.forward = _actor_forward  # type: ignore[method-assign]
    policy.exploration_noise = _expl_fn  # type: ignore[method-assign]
    # Training: resample bin y on bin x (3.5) each episode reset.
    train_envs.set_env_attr("randomize_obstacle", True)
    train_envs.set_env_attr("include_middle_pass", False)
    print(
        "[INFO] Initial waypoint collection done; restoring actor policy "
        "(train: a_nom & SF yaw=0; goal=random left|right of bin). "
        "Obstacle randomization ON (x=3.5, y∈[-0.5,0.5]); y_bound disabled."
    )

    # Critic-only warm-up on buffer (policy.warmup=True → actor loss skipped).
    n_critic_wu = int(getattr(args, "critic_warmup_updates", 1000))
    if n_critic_wu > 0 and not args.resume_policy:
        print(f"[INFO] Critic-only warmup: {n_critic_wu} updates (actor frozen)...")
        policy.warmup = True
        for i in range(1, n_critic_wu + 1):
            metrics = policy.update(args.batch_size_pyhj, buffer)
            env_step = int(log_state.get("env_step", 0))
            log_state["update"] = int(log_state.get("update", 0)) + 1
            wandb.log(
                {
                    "trainer/env_step": float(env_step),
                    "trainer/update": float(log_state["update"]),
                    "loss/actor": float(metrics.get("loss/actor", 0.0)),
                    "loss/critic": float(metrics["loss/critic"]),
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

    # Optional: BC actor to waypoint buffer acts before joint RL.
    n_actor_bc = int(getattr(args, "actor_bc_warmup_updates", 0))
    if n_actor_bc > 0 and not args.resume_policy:
        print(
            f"[INFO] Actor BC warmup: {n_actor_bc} updates "
            f"(imitate waypoint act_nom / buffer acts)..."
        )
        for i in range(1, n_actor_bc + 1):
            batch, _ = buffer.sample(args.batch_size_pyhj)
            act_pred = policy(batch, model="actor").act
            act_demo = _batch_act_nom_tensor(batch, act_pred)
            bc_loss = torch.nn.functional.mse_loss(act_pred, act_demo)
            policy.actor_optim.zero_grad()
            bc_loss.backward()
            policy.actor_optim.step()
            policy.actor_old.load_state_dict(policy.actor.state_dict())
            env_step = int(log_state.get("env_step", 0))
            log_state["update"] = int(log_state.get("update", 0)) + 1
            wandb.log(
                {
                    "trainer/env_step": float(env_step),
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
        print("[INFO] Actor BC warmup done; enabling joint actor-critic training.")

    log_path = Path(f"runs/ddpg_hj_humanoid/{args.dino_encoder}-{timestamp}")
    if args.resume_policy:
        log_path = Path(
            f"runs/ddpg_hj_humanoid/{args.dino_encoder}-{timestamp}-resume{resume_epoch}"
        )
    print(
        f"[INFO] Training epochs {start_epoch}..{end_epoch}; ckpts -> {log_path}"
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
