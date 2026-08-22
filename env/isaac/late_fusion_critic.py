"""Q(z, a) with late action fusion.

Early concat of a ~70k-D latent z with a 3-D action makes the first Linear
ignore a, so ∇_a Q ≈ 0 and the SAC actor mean collapses. Encode z and a
separately to the same width, then fuse.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn

from PyHJ.utils.net.common import MLP


def _flat_dim(shape) -> int:
    if isinstance(shape, int):
        return int(shape)
    return int(np.prod(shape))


class LateFusionCritic(nn.Module):
    """Mapping: z → MLP, a → MLP, cat → MLP → Q.

    Action is lifted to the same width as the z feature before concat, so it
    is not a 3-vs-512 skip at the fusion layer.

    Interface matches ``PyHJ.utils.net.continuous.Critic``:
    ``forward(obs, act) -> (B, 1)``.
    """

    def __init__(
        self,
        state_shape,
        action_shape,
        hidden_sizes: Sequence[int] = (512, 512, 512),
        activation: type[nn.Module] = nn.ReLU,
        device: Union[str, int, torch.device] = "cpu",
    ) -> None:
        super().__init__()
        hidden = tuple(int(h) for h in hidden_sizes)
        if not hidden:
            raise ValueError("hidden_sizes must be non-empty")
        self.device = device
        self.obs_dim = _flat_dim(state_shape)
        self.act_dim = _flat_dim(action_shape)
        self.output_dim = 1
        feat_dim = hidden[-1]
        self.feat_dim = feat_dim
        # z only: obs_dim → hidden[0] → … → hidden[-1]
        self.z_mlp = MLP(
            self.obs_dim,
            output_dim=0,
            hidden_sizes=hidden,
            activation=activation,
            device=device,
        )
        # a: 3 → feat/4 → feat, so fusion is feat || feat (equal width)
        act_hidden = (max(feat_dim // 4, 32), feat_dim)
        self.a_mlp = MLP(
            self.act_dim,
            output_dim=0,
            hidden_sizes=act_hidden,
            activation=activation,
            device=device,
        )
        self.q_head = MLP(
            feat_dim * 2,
            output_dim=1,
            hidden_sizes=(feat_dim,),
            activation=activation,
            device=device,
        )

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32).flatten(1)
        z_feat = self.z_mlp(obs)
        if act is None:
            raise ValueError("LateFusionCritic requires act; Q(z, a) is action-conditioned")
        act = torch.as_tensor(act, device=self.device, dtype=torch.float32).flatten(1)
        if act.shape[0] != z_feat.shape[0]:
            raise ValueError(f"batch mismatch: z {z_feat.shape} vs a {act.shape}")
        if act.shape[-1] != self.act_dim:
            raise ValueError(f"expected act dim {self.act_dim}, got {act.shape[-1]}")
        a_feat = self.a_mlp(act)
        return self.q_head(torch.cat([z_feat, a_feat], dim=1))


def make_concat_critic(
    state_shape,
    action_shape,
    hidden_sizes: Sequence[int],
    activation: type[nn.Module],
    device: Union[str, int, torch.device],
):
    """Q(cat(z, a)): first Linear sees z and the 3-D action together.

    Intended for short z (DINOv2 CLS, 384-D). Do not use with ~70k patch z.
    """
    from PyHJ.utils.net.common import Net
    from PyHJ.utils.net.continuous import Critic

    net = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        activation=activation,
        concat=True,
        device=device,
    )
    critic = Critic(net, device=device).to(device)
    obs_dim = _flat_dim(state_shape)
    act_dim = _flat_dim(action_shape)
    hid = ",".join(str(h) for h in hidden_sizes)
    print(
        f"[INFO] ConcatCritic: cat(z({obs_dim}), a({act_dim})) → MLP[{hid}] → Q"
    )
    return critic


def make_q_critic(
    fusion: str,
    state_shape,
    action_shape,
    hidden_sizes: Sequence[int],
    activation: type[nn.Module],
    device: Union[str, int, torch.device],
):
    fusion = str(fusion).lower()
    if fusion == "late":
        return make_late_fusion_critic(
            state_shape, action_shape, hidden_sizes, activation, device
        )
    if fusion == "concat":
        return make_concat_critic(
            state_shape, action_shape, hidden_sizes, activation, device
        )
    raise ValueError(f"Unknown critic_fusion={fusion!r}; use 'late' or 'concat'")


def make_late_fusion_critic(
    state_shape,
    action_shape,
    hidden_sizes: Sequence[int],
    activation: type[nn.Module],
    device: Union[str, int, torch.device],
) -> LateFusionCritic:
    critic = LateFusionCritic(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        activation=activation,
        device=device,
    ).to(device)
    obs_dim = _flat_dim(state_shape)
    act_dim = _flat_dim(action_shape)
    hid = ",".join(str(h) for h in hidden_sizes)
    feat = int(hidden_sizes[-1])
    print(
        f"[INFO] LateFusionCritic: z({obs_dim}) → MLP[{hid}]={feat}, "
        f"a({act_dim}) → MLP[{feat // 4},{feat}]={feat}, cat → Q"
    )
    return critic
