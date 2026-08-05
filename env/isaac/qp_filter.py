"""Discrete QP-style safety filter: closest action to nominal with Q >= threshold.

Mirrors CarGoal ``mode == "QP"``; humanoid actions are (vx, vy, yaw) in policy space.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


@torch.no_grad()
def qp_filter_action_policy(
    critic: torch.nn.Module,
    obs: Any,
    a_nom_policy: np.ndarray | torch.Tensor,
    *,
    safety_threshold: float = 0.0,
    n_grid: int = 21,
    freeze_yaw: bool = False,
    yaw: float = 0.0,
    device: str | torch.device | None = None,
    chunk_size: int = 256,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return policy-space action (3,) and debug info.

    If ``Q(z, a_nom) >= threshold``, return ``a_nom``.
    Else search a discrete grid in [-1, 1]:
      - ``freeze_yaw=False``: ``n_grid³`` on (vx, vy, yaw)
      - ``freeze_yaw=True``: ``n_grid²`` on (vx, vy), yaw fixed to ``yaw``
    Among candidates with Q >= threshold pick closest to ``a_nom`` (L2).
    If none are safe, fall back to argmax_Q on the grid.
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
    if obs_t.shape[0] != 1:
        raise ValueError(f"qp_filter expects a single state, got batch={obs_t.shape[0]}")

    a_nom = np.asarray(a_nom_policy, dtype=np.float32).reshape(-1)
    if a_nom.size < 2:
        raise ValueError(f"a_nom must have at least vx,vy, got shape {a_nom.shape}")
    a_nom_3 = np.zeros(3, dtype=np.float32)
    a_nom_3[0] = float(np.clip(a_nom[0], -1.0, 1.0))
    a_nom_3[1] = float(np.clip(a_nom[1], -1.0, 1.0))
    if freeze_yaw:
        a_nom_3[2] = float(yaw)
    else:
        a_nom_3[2] = float(np.clip(a_nom[2], -1.0, 1.0)) if a_nom.size >= 3 else 0.0

    a_nom_t = torch.as_tensor(a_nom_3, dtype=torch.float32, device=device).unsqueeze(0)
    q_nom = float(critic(obs_t, a_nom_t).reshape(-1)[0].item())
    info: dict[str, Any] = {
        "q_nom": q_nom,
        "intervened": False,
        "n_safe": 0,
        "fallback_maxq": False,
        "freeze_yaw": bool(freeze_yaw),
    }
    if q_nom >= safety_threshold:
        return a_nom_3, info

    info["intervened"] = True
    g = torch.linspace(-1.0, 1.0, int(n_grid), device=device)
    if freeze_yaw:
        vx, vy = torch.meshgrid(g, g, indexing="ij")
        grid = torch.stack(
            [
                vx.reshape(-1),
                vy.reshape(-1),
                torch.full((vx.numel(),), float(yaw), device=device),
            ],
            dim=-1,
        )
    else:
        vx, vy, yaw_g = torch.meshgrid(g, g, g, indexing="ij")
        grid = torch.stack(
            [vx.reshape(-1), vy.reshape(-1), yaw_g.reshape(-1)],
            dim=-1,
        )

    q_all = torch.empty(grid.shape[0], device=device, dtype=torch.float32)
    obs_rep_base = obs_t.expand(chunk_size, -1)
    for start in range(0, grid.shape[0], chunk_size):
        end = min(start + chunk_size, grid.shape[0])
        n = end - start
        q_all[start:end] = critic(obs_rep_base[:n], grid[start:end]).reshape(-1)

    safe_mask = q_all >= safety_threshold
    n_safe = int(safe_mask.sum().item())
    info["n_safe"] = n_safe
    a_nom_t3 = a_nom_t.reshape(1, 3)
    if n_safe > 0:
        safe_acts = grid[safe_mask]
        dist = torch.norm(safe_acts - a_nom_t3, dim=-1)
        best = safe_acts[int(dist.argmin().item())]
    else:
        info["fallback_maxq"] = True
        best = grid[int(q_all.argmax().item())]

    out = best.detach().cpu().numpy().astype(np.float32).reshape(3)
    if freeze_yaw:
        out[2] = float(yaw)
    info["q_chosen"] = float(critic(obs_t, best.unsqueeze(0)).reshape(-1)[0].item())
    return out, info
