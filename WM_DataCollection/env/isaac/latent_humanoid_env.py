"""Gymnasium env that encodes Isaac G1 observations with a DINO world model."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box

from env.isaac.isaac_g1_wrapper import IsaacG1Wrapper


class LatentHumanoidEnv(gym.Env):
    """Latent-space G1 env for PyHJ avoid-DDPG safety-filter training."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        args,
        wm,
        device: str,
        args_cli,
        with_proprio: bool = False,
        latent_h: bool = False,
    ):
        super().__init__()
        self.args = args
        self.device = torch.device(device)
        self.with_proprio = with_proprio
        self.latent_h = latent_h
        self.wm = wm
        self.wm.eval()

        self.wrapper = IsaacG1Wrapper(
            args_cli,
            enable_cameras=getattr(args_cli, "enable_cameras", True),
        )

        if latent_h:
            raise NotImplementedError("FailureClassifier latent_h is not wired for Isaac G1 yet.")

        reset_info = self.wrapper.reset_scene(seed=getattr(args, "seed", None))
        obs = self.wrapper.get_raw_obs()
        z = self.encode(obs)
        print(f"[LatentHumanoidEnv] latent shape: {z.shape}, reset: {reset_info}")

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=z.shape, dtype=np.float32
        )
        self.action_space = Box(
            low=np.array([-0.5, -0.5, -1.0], dtype=np.float32),
            high=np.array([0.5, 0.5, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        reset_info = self.wrapper.reset_scene(seed=seed)
        obs = self.wrapper.get_raw_obs()
        z = self.encode(obs)
        info = {"state": obs, **reset_info}
        return z, info

    def step(self, action):
        _, terminated, truncated, step_info = self.wrapper.apply_velocity_command(action)
        obs = self.wrapper.get_raw_obs()
        h_s = self.wrapper.calculate_cost()
        z_next = self.encode(obs)
        info = {"state": obs, **step_info}
        return z_next, h_s, terminated, truncated, info

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
