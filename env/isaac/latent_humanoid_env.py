"""Gymnasium env that encodes Isaac G1 observations with a DINO world model."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box

from env.isaac.isaac_g1_wrapper import IsaacG1Wrapper

# Match datasets/humanoid_dset.py and collect_humanoid_dataset.py
DEFAULT_VISUAL_FPS = 15.0
DEFAULT_MAX_EPISODE_SIM_STEPS = 3000


class LatentHumanoidEnv(gym.Env):
    """Latent-space G1 env for PyHJ avoid-DDPG safety-filter training.

    Time alignment (same as offline WM data):
    - Physics / velocity commands run at sim rate (~200 Hz).
    - One Gym/HJ step = hold the same action until the next 15 fps visual
      sample time, then encode visual+proprio once (1:1 with WM timeline).

    Episode ends (not based on safety cost ``h_s``):
    - all waypoints reached
    - stuck contact on a non-ankle_roll link for ``stuck_contact_steps``
    - sim-step count hits ``max_episode_steps`` (default 3000 control steps)

    Collision / LiDAR ``h_s`` is the step cost but does **not** end the episode.
    """

    metadata = {"render_modes": []}

    # Fixed info schema for PyHJ Batch (reset/step must use the same keys).
    # 0=ongoing, 1=all_waypoints, 2=stuck, 3=max_steps
    _END_REASON_CODE = {
        None: 0,
        "all_waypoints": 1,
        "stuck": 2,
        "max_steps": 3,
    }

    def __init__(
        self,
        args,
        wm,
        device: str,
        args_cli,
        with_proprio: bool = False,
        latent_h: bool = False,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_SIM_STEPS,
        visual_fps: float = DEFAULT_VISUAL_FPS,
    ):
        super().__init__()
        self.args = args
        self.device = torch.device(device)
        self.with_proprio = with_proprio
        self.latent_h = latent_h
        self.wm = wm
        self.wm.eval()
        self.max_episode_sim_steps = int(
            getattr(args, "max_episode_steps", max_episode_steps)
        )
        self.visual_fps = float(getattr(args, "visual_fps", visual_fps))
        self.visual_period_s = 1.0 / self.visual_fps

        self._episode_sim_step = 0
        self._episode_visual_step = 0
        self._next_visual_time_s = 0.0

        # enable_cameras is handled by AppLauncher / visual_mode on args_cli,
        # not as an IsaacG1Wrapper kwarg.
        self.wrapper = IsaacG1Wrapper(args_cli)
        self.sim_dt = float(self.wrapper.sim_dt)

        if latent_h:
            raise NotImplementedError("FailureClassifier latent_h is not wired for Isaac G1 yet.")

        reset_info = self.wrapper.reset_scene(seed=getattr(args, "seed", None))
        self._reset_timers()
        obs = self.wrapper.get_raw_obs()
        z = self.encode(obs)
        approx_substeps = max(1, int(round(self.visual_period_s / self.sim_dt)))
        print(
            f"[LatentHumanoidEnv] latent shape: {z.shape}, "
            f"sim_dt={self.sim_dt:.6f} (~{1.0 / self.sim_dt:.1f} Hz), "
            f"visual_fps={self.visual_fps}, ~{approx_substeps} sim steps / HJ step, "
            f"max_episode_sim_steps={self.max_episode_sim_steps}, reset: {reset_info}"
        )

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=z.shape, dtype=np.float32
        )
        self.action_space = Box(
            low=np.array([-0.5, -0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 0.5, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def _reset_timers(self) -> None:
        self._episode_sim_step = 0
        self._episode_visual_step = 0
        # Next encode boundary after t=0 (reset already encoded the t=0 frame).
        self._next_visual_time_s = self.visual_period_s

    def _pyhj_info(self, end_reason: str | None = None, stuck: bool = False) -> dict:
        """Scalar-only info with a fixed key set (required by PyHJ Batch assignment)."""
        return {
            "episode_step": np.int32(self._episode_visual_step),
            "end_reason": np.int32(self._END_REASON_CODE.get(end_reason, 0)),
            "stuck": np.float32(1.0 if stuck else 0.0),
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        reset_info = self.wrapper.reset_scene(seed=seed)
        self._reset_timers()
        obs = self.wrapper.get_raw_obs()
        z = self.encode(obs)
        if reset_info is not None:
            print(
                f"[LatentHumanoidEnv] reset robot_xy={reset_info.get('robot_xy')} "
                f"waypoints={reset_info.get('waypoints')}"
            )
        return z, self._pyhj_info(end_reason=None, stuck=False)

    def step(self, action):
        """Hold ``action`` across sim substeps until the next 15 fps visual sample."""
        h_s = 0.0
        stuck = False
        end_reason = None
        step_info: dict[str, Any] = {}

        # Run control-rate physics until we hit the next visual sample time
        # (same gating idea as collect_humanoid_dataset.subsample_frame_indices).
        while True:
            _, _isaac_terminated, _, step_info = self.wrapper.apply_velocity_command(action)
            del _isaac_terminated
            self._episode_sim_step += 1
            sim_time_s = self._episode_sim_step * self.sim_dt

            h_s = max(h_s, float(self.wrapper.calculate_cost()))
            stuck = stuck or bool(step_info.get("stuck", False))

            if self.wrapper.advance_waypoint_if_reached():
                end_reason = "all_waypoints"
                break
            if stuck:
                end_reason = "stuck"
                break
            if self._episode_sim_step >= self.max_episode_sim_steps:
                end_reason = "max_steps"
                break

            if sim_time_s + 1e-12 >= self._next_visual_time_s:
                self._next_visual_time_s += self.visual_period_s
                break

        self._episode_visual_step += 1
        obs = self.wrapper.get_raw_obs()
        z_next = self.encode(obs)

        terminated = False
        truncated = end_reason is not None
        if truncated:
            print(
                f"[LatentHumanoidEnv] episode end reason={end_reason} "
                f"visual_steps={self._episode_visual_step} "
                f"sim_steps={self._episode_sim_step}"
            )
        return z_next, h_s, terminated, truncated, self._pyhj_info(end_reason, stuck)

    def encode(self, obs: dict[str, Any] | tuple | list) -> np.ndarray:
        """Encode visual + proprio into a flat latent vector via the world model."""
        if isinstance(obs, dict):
            visual = obs["visual"]
            proprio = obs["proprio"]
        elif isinstance(obs, (tuple, list)) and len(obs) == 2:
            visual, proprio = obs
        else:
            raise ValueError(f"Unexpected obs type: {type(obs)}")

        with torch.no_grad():
            if isinstance(visual, torch.Tensor):
                visual_np = visual.permute(2, 0, 1).float().cpu().numpy()
                if visual_np.max() > 1.0:
                    visual_np /= 255.0
                visual_np = (visual_np - 0.5) / 0.5
                vis_t = torch.from_numpy(visual_np).unsqueeze(0).unsqueeze(1).to(self.device)
                prop_t = proprio.unsqueeze(0).unsqueeze(1).float().to(self.device)
            else:
                visual_np = np.transpose(visual, (2, 0, 1)).astype(np.float32)
                visual_np /= 255.0
                visual_np = (visual_np - 0.5) / 0.5
                vis_t = torch.from_numpy(visual_np).unsqueeze(0).unsqueeze(1).to(self.device)
                prop_t = (
                    torch.from_numpy(np.asarray(proprio, dtype=np.float32))
                    .unsqueeze(0)
                    .unsqueeze(1)
                    .to(self.device)
                )

            lat = self.wm.encode_obs({"visual": vis_t, "proprio": prop_t})

            if self.with_proprio:
                z_vis = lat["visual"].reshape(1, -1)
                z_prop = lat["proprio"].squeeze(0)
                z = torch.cat([z_vis, z_prop], dim=-1)
                return z.squeeze(0).cpu().numpy()

            z_vis = lat["visual"].reshape(1, -1)
            return z_vis.squeeze(0).cpu().numpy()

    def calculate_cost(self) -> float:
        return self.wrapper.calculate_cost()

    def close(self):
        self.wrapper.close()
