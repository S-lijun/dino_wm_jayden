"""Select actions by maximizing a critic over sampled candidates (no actor)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


@torch.no_grad()
def select_actions_max_q(
    critic: torch.nn.Module,
    obs: Any,
    *,
    act_dim: int = 3,
    n_candidates: int = 64,
    device: str | torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each state, sample actions in [-1, 1] and pick argmax_a Q(s, a).

    Returns
    -------
    best_act : (B, act_dim) float tensor in [-1, 1]
    best_q   : (B,) float tensor
    """
    if device is None:
        try:
            device = next(critic.parameters()).device
        except StopIteration:
            device = "cpu"
    if not torch.is_tensor(obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    else:
        obs_t = obs.to(device=device, dtype=torch.float32)
    if obs_t.ndim == 1:
        obs_t = obs_t.unsqueeze(0)

    batch_size = int(obs_t.shape[0])
    # Uniform candidates in policy space [-1, 1].
    cands = torch.empty(
        batch_size, n_candidates, act_dim, device=obs_t.device, dtype=torch.float32
    ).uniform_(-1.0, 1.0)
    # Always include the zero action (often a safe baseline).
    cands[:, 0, :] = 0.0

    obs_exp = (
        obs_t.unsqueeze(1)
        .expand(batch_size, n_candidates, obs_t.shape[-1])
        .reshape(batch_size * n_candidates, -1)
    )
    act_flat = cands.reshape(batch_size * n_candidates, act_dim)
    q = critic(obs_exp, act_flat).reshape(batch_size, n_candidates)
    best_idx = q.argmax(dim=1)
    arange = torch.arange(batch_size, device=obs_t.device)
    best_act = cands[arange, best_idx]
    best_q = q[arange, best_idx]
    return best_act, best_q


@torch.no_grad()
def select_action_max_q_numpy(
    critic: torch.nn.Module,
    obs: np.ndarray,
    *,
    act_dim: int = 3,
    n_candidates: int = 64,
    device: str | torch.device | None = None,
) -> tuple[np.ndarray, float]:
    """Single-state helper → (act_policy[-1,1], q)."""
    best_act, best_q = select_actions_max_q(
        critic,
        obs,
        act_dim=act_dim,
        n_candidates=n_candidates,
        device=device,
    )
    return (
        best_act[0].detach().cpu().numpy().astype(np.float32).reshape(-1),
        float(best_q[0].item()),
    )
