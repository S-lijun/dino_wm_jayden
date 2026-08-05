"""Select actions by maximizing a critic over sampled candidates (no actor)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import torch


def _apply_fixed_action_dims(
    acts: torch.Tensor,
    fixed_dims: Mapping[int, float] | None,
) -> torch.Tensor:
    """In-place-safe: set selected action dims to constants (policy space)."""
    if not fixed_dims:
        return acts
    out = acts
    for dim, value in fixed_dims.items():
        d = int(dim)
        if d < 0 or d >= out.shape[-1]:
            raise ValueError(
                f"fixed action dim {d} out of range for act_dim={out.shape[-1]}"
            )
        out[..., d] = float(value)
    return out


@torch.no_grad()
def select_actions_max_q(
    critic: torch.nn.Module,
    obs: Any,
    *,
    act_dim: int = 3,
    n_candidates: int = 64,
    device: str | torch.device | None = None,
    chunk_size: int = 16,
    fixed_dims: Mapping[int, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """For each state, sample actions in [-1, 1] and pick argmax_a Q(s, a).

    Candidates are scored in chunks so peak VRAM stays O(B * chunk_size) instead
    of O(B * n_candidates) — important next to Isaac RTX on 24GB GPUs.

    ``fixed_dims`` forces selected action dimensions to constants for every
    candidate (e.g. ``{2: 0.0}`` freezes yaw in policy space).

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
    elif obs_t.ndim != 2:
        raise ValueError(
            f"select_actions_max_q expects obs shape (dim,) or (B, dim), got {tuple(obs_t.shape)}"
        )

    batch_size = int(obs_t.shape[0])
    n_candidates = int(n_candidates)
    chunk_size = max(1, int(chunk_size))

    # Uniform candidates in policy space [-1, 1].
    cands = torch.empty(
        batch_size, n_candidates, act_dim, device=obs_t.device, dtype=torch.float32
    ).uniform_(-1.0, 1.0)
    # Always include the zero action (often a safe baseline).
    cands[:, 0, :] = 0.0
    cands = _apply_fixed_action_dims(cands, fixed_dims)

    best_q = torch.full(
        (batch_size,), -float("inf"), device=obs_t.device, dtype=torch.float32
    )
    best_act = torch.zeros(
        batch_size, act_dim, device=obs_t.device, dtype=torch.float32
    )
    best_act = _apply_fixed_action_dims(best_act, fixed_dims)

    for start in range(0, n_candidates, chunk_size):
        end = min(start + chunk_size, n_candidates)
        n_c = end - start
        act_chunk = cands[:, start:end, :].reshape(batch_size * n_c, act_dim)
        obs_exp = (
            obs_t.unsqueeze(1)
            .expand(batch_size, n_c, obs_t.shape[-1])
            .reshape(batch_size * n_c, -1)
        )
        q_chunk = critic(obs_exp, act_chunk).reshape(batch_size, n_c)
        chunk_best_q, chunk_best_idx = q_chunk.max(dim=1)
        replace = chunk_best_q > best_q
        if replace.any():
            best_q = torch.where(replace, chunk_best_q, best_q)
            arange = torch.arange(batch_size, device=obs_t.device)
            chosen = cands[arange, start + chunk_best_idx]
            best_act = torch.where(replace.unsqueeze(-1), chosen, best_act)

    best_act = _apply_fixed_action_dims(best_act, fixed_dims)
    return best_act, best_q


@torch.no_grad()
def select_action_max_q_numpy(
    critic: torch.nn.Module,
    obs: np.ndarray,
    *,
    act_dim: int = 3,
    n_candidates: int = 64,
    device: str | torch.device | None = None,
    chunk_size: int = 16,
    fixed_dims: Mapping[int, float] | None = None,
) -> tuple[np.ndarray, float]:
    """Single-state helper → (act_policy[-1,1], q)."""
    best_act, best_q = select_actions_max_q(
        critic,
        obs,
        act_dim=act_dim,
        n_candidates=n_candidates,
        device=device,
        chunk_size=chunk_size,
        fixed_dims=fixed_dims,
    )
    return (
        best_act[0].detach().cpu().numpy().astype(np.float32).reshape(-1),
        float(best_q[0].item()),
    )
