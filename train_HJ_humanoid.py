"""DDPG HJ safety-filter training on Isaac G1 latent space."""

import argparse
import os
import sys
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
    default=5,
    help="Upload a rollout video to wandb every N finished episodes (0 disables).",
)
# --device is already registered by AppLauncher.add_app_launcher_args()

args_cli, remaining = parser.parse_known_args()

sys.path.insert(0, os.path.join(ISAACLAB_ROOT, "scripts/demos"))
from visual_obs_utils import configure_app_for_visual, resolve_visual_mode

_visual_mode = resolve_visual_mode(args_cli)
configure_app_for_visual(args_cli, _visual_mode)
torch_device = getattr(args_cli, "device", "cuda:0")
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

from PyHJ.data import Collector, VectorReplayBuffer
from PyHJ.trainer import offpolicy_trainer
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.utils import WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
from PyHJ.policy import avoid_DDPGPolicy_annealing

from wm_load import load_model  # not plan.py — avoids env.venv / utils collision
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
    writer = SummaryWriter(log_dir=f"runs/ddpg_hj_humanoid/{args.dino_encoder}-{timestamp}/logs")
    wb_logger = WandbLogger()
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
    max_action = torch.tensor(action_space.high, device=args.device, dtype=torch.float32)

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
        action_space=action_space,
        actor=actor,
        actor_optim=actor_optim,
        actor_gradient_steps=args.actor_gradient_steps,
    )

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

    orig_learn = policy.learn
    policy.last_actor_loss = 0.0
    policy.last_critic_loss = 0.0

    def learn_and_record(batch, **kw):
        metrics = orig_learn(batch, **kw)
        policy.last_actor_loss = metrics["loss/actor"]
        policy.last_critic_loss = metrics["loss/critic"]
        return metrics

    policy.learn = learn_and_record

    def train_fn(epoch: int, step_idx: int):
        wandb.log(
            {
                "loss/actor": policy.last_actor_loss,
                "loss/critic": policy.last_critic_loss,
            }
        )

    buffer = VectorReplayBuffer(args.buffer_size, args.training_num)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    # Replay buffer is not restored; collect a fresh warm-up even when resuming weights.
    print("[INFO] Collecting initial transitions...")
    train_collector.collect(1000)
    print("[INFO] Initial collection done.")

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

        numeric = {}
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                numeric[f"train/{k}"] = v
            elif isinstance(v, np.generic):
                numeric[f"train/{k}"] = float(v)
        wandb.log(numeric, step=epoch)

        ckpt_dir_epoch = log_path / f"epoch_id_{epoch}"
        ckpt_dir_epoch.mkdir(exist_ok=True, parents=True)
        torch.save(policy.state_dict(), ckpt_dir_epoch / "policy.pth")

    print("[INFO] Training complete.")
    simulation_app.close()


if __name__ == "__main__":
    main()
