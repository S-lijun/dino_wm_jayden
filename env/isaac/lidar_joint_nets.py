"""Dual-stream joints + lidar encoder, late-fused with action.

Raw concat of 12 joint angles with ~2700 lidar ranges lets the first Linear
ignore joints. Encode each stream to the same width, cat to z, then lift
the 3-D action to that z width before the Q head.

Actor uses the same dual-stream z so the policy is not lidar-only either.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn

from PyHJ.utils.net.common import MLP

# Matches env.isaac.isaac_g1_wrapper.G1_LOWER_BODY_JOINTS (12-D legs).
DEFAULT_JOINT_DIM = 12


def _flat_dim(shape) -> int:
    if isinstance(shape, int):
        return int(shape)
    return int(np.prod(shape))


def _as_bflat(x, device) -> torch.Tensor:
    t = torch.as_tensor(x, device=device, dtype=torch.float32)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t.flatten(1)


class DualStreamEncoder(nn.Module):
    """obs = joints || lidar → 256 || 256 = 512 z.

    ``forward`` matches Tianshou/PyHJ ``Net``: ``(obs, state, info) -> (z, state)``.
    """

    def __init__(
        self,
        state_shape,
        *,
        joint_dim: int = DEFAULT_JOINT_DIM,
        hidden_sizes: Sequence[int] = (256,),
        activation: type[nn.Module] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        hidden = tuple(int(h) for h in hidden_sizes)
        if not hidden:
            raise ValueError("hidden_sizes must be non-empty")
        self.device = device
        self.obs_dim = _flat_dim(state_shape)
        self.joint_dim = int(joint_dim)
        if self.obs_dim <= self.joint_dim:
            raise ValueError(
                f"obs_dim={self.obs_dim} must be > joint_dim={self.joint_dim}"
            )
        self.lidar_dim = self.obs_dim - self.joint_dim
        stream_dim = hidden[-1]
        self.stream_dim = stream_dim
        self.output_dim = stream_dim * 2
        self.joint_mlp = MLP(
            self.joint_dim,
            output_dim=0,
            hidden_sizes=(max(stream_dim // 2, 64), stream_dim),
            activation=activation,
            device=device,
        )
        self.lidar_mlp = MLP(
            self.lidar_dim,
            output_dim=0,
            hidden_sizes=hidden,
            activation=activation,
            device=device,
        )

    def encode_z(self, obs) -> torch.Tensor:
        obs = _as_bflat(obs, self.device)
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"expected obs dim {self.obs_dim}, got {obs.shape[-1]}")
        j = obs[:, : self.joint_dim]
        lidar = obs[:, self.joint_dim :]
        return torch.cat([self.joint_mlp(j), self.lidar_mlp(lidar)], dim=1)

    def forward(self, obs, state=None, info: Dict[str, Any] | None = None, **kwargs):
        return self.encode_z(obs), state


class DualStreamLateFusionCritic(nn.Module):
    """Q(joints, lidar, a): joints→256, lidar→256, z=512, a→512, cat→Q.

    Interface matches ``PyHJ.utils.net.continuous.Critic``:
    ``forward(obs, act) -> (B, 1)``.
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        hidden_sizes: Sequence[int] = (256,),
        activation: type[nn.Module] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
        joint_dim: int = DEFAULT_JOINT_DIM,
    ) -> None:
        super().__init__()
        self.device = device
        self.encoder = DualStreamEncoder(
            state_shape,
            joint_dim=joint_dim,
            hidden_sizes=hidden_sizes,
            activation=activation,
            device=device,
        )
        self.act_dim = _flat_dim(action_shape)
        self.output_dim = 1
        z_dim = self.encoder.output_dim
        self.feat_dim = z_dim
        self.a_mlp = MLP(
            self.act_dim,
            output_dim=0,
            hidden_sizes=(max(z_dim // 4, 32), z_dim),
            activation=activation,
            device=device,
        )
        self.q_head = MLP(
            z_dim * 2,
            output_dim=1,
            hidden_sizes=(z_dim,),
            activation=activation,
            device=device,
        )

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        z_feat = self.encoder.encode_z(obs)
        if act is None:
            raise ValueError(
                "DualStreamLateFusionCritic requires act; Q(z, a) is action-conditioned"
            )
        act = _as_bflat(act, self.device)
        if act.shape[0] != z_feat.shape[0]:
            raise ValueError(f"batch mismatch: z {z_feat.shape} vs a {act.shape}")
        if act.shape[-1] != self.act_dim:
            raise ValueError(f"expected act dim {self.act_dim}, got {act.shape[-1]}")
        a_feat = self.a_mlp(act)
        return self.q_head(torch.cat([z_feat, a_feat], dim=1))


def make_dual_stream_critic(
    state_shape,
    action_shape,
    hidden_sizes: Sequence[int],
    activation: type[nn.Module],
    device: Union[str, int, torch.device],
    joint_dim: int = DEFAULT_JOINT_DIM,
) -> DualStreamLateFusionCritic:
    critic = DualStreamLateFusionCritic(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        activation=activation,
        device=device,
        joint_dim=joint_dim,
    ).to(device)
    enc = critic.encoder
    hid = ",".join(str(h) for h in hidden_sizes)
    print(
        f"[INFO] DualStreamLateFusionCritic: "
        f"joints({enc.joint_dim})→{enc.stream_dim}, "
        f"lidar({enc.lidar_dim})→MLP[{hid}]={enc.stream_dim}, "
        f"z={enc.output_dim}; "
        f"a({critic.act_dim})→{enc.output_dim}; cat(z,a)={enc.output_dim * 2}→Q"
    )
    return critic


def make_dual_stream_actor_net(
    state_shape,
    hidden_sizes: Sequence[int],
    activation: type[nn.Module],
    device: Union[str, int, torch.device],
    joint_dim: int = DEFAULT_JOINT_DIM,
) -> DualStreamEncoder:
    net = DualStreamEncoder(
        state_shape,
        joint_dim=joint_dim,
        hidden_sizes=hidden_sizes,
        activation=activation,
        device=device,
    ).to(device)
    hid = ",".join(str(h) for h in hidden_sizes)
    print(
        f"[INFO] DualStreamActorEncoder: "
        f"joints({net.joint_dim})→{net.stream_dim}, "
        f"lidar({net.lidar_dim})→MLP[{hid}]={net.stream_dim}, "
        f"z={net.output_dim}"
    )
    return net
