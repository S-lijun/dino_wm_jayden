#!/usr/bin/env python3
"""Merge multiple humanoid_g1 collection sessions into one training dataset."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch

OBS_SRC = "obses_15fps"
OBS_DST = "obses"


def _load(path: Path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def merge_sessions(
    base_dir: Path,
    sessions: list[str],
    output_dir: Path | None = None,
    obs_src: str = OBS_SRC,
    obs_dst: str = OBS_DST,
) -> None:
    base_dir = Path(base_dir)
    out_dir = Path(output_dir) if output_dir else base_dir / "session_dir"
    episodes_actions: list[torch.Tensor] = []
    episodes_states: list[torch.Tensor] = []
    episodes_costs: list[torch.Tensor] = []
    flat_seq_lengths: list[int] = []
    metas: list[dict] = []

    for session in sessions:
        session_dir = base_dir / session
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Missing session directory: {session_dir}")

        actions = _load(session_dir / "actions.pth").float()
        states = _load(session_dir / "states.pth").float()
        costs = _load(session_dir / "costs.pth").float()
        seq_lengths = _load(session_dir / "seq_lengths.pth").long()
        meta = _load(session_dir / "meta.pth")
        obs_dir = session_dir / obs_src
        if not obs_dir.is_dir():
            raise FileNotFoundError(f"Missing observation directory: {obs_dir}")

        n_eps = len(seq_lengths)
        for i in range(n_eps):
            obs_file = obs_dir / f"episode_{i}.pth"
            if not obs_file.exists():
                raise FileNotFoundError(f"Missing observation file: {obs_file}")

        for i in range(n_eps):
            t = int(seq_lengths[i].item())
            episodes_actions.append(actions[i, :t])
            episodes_states.append(states[i, :t])
            episodes_costs.append(costs[i, :t])
            flat_seq_lengths.append(t)

        metas.append(meta)
        print(
            f"[{session}] episodes={n_eps}, "
            f"steps={int(seq_lengths.sum().item())}, "
            f"max_len={int(seq_lengths.max().item())}"
        )

    max_len = max(flat_seq_lengths)
    n_total = len(flat_seq_lengths)
    action_dim = episodes_actions[0].shape[-1]
    state_dim = episodes_states[0].shape[-1]

    padded_actions = torch.zeros(n_total, max_len, action_dim)
    padded_states = torch.zeros(n_total, max_len, state_dim)
    padded_costs = torch.zeros(n_total, max_len)

    for i, (action, state, cost) in enumerate(
        zip(episodes_actions, episodes_states, episodes_costs)
    ):
        t = flat_seq_lengths[i]
        padded_actions[i, :t] = action
        padded_states[i, :t] = state
        padded_costs[i, :t] = cost

    obs_out = out_dir / obs_dst
    if obs_out.exists():
        shutil.rmtree(obs_out)
    obs_out.mkdir(parents=True, exist_ok=True)

    global_idx = 0
    for session in sessions:
        session_dir = base_dir / session
        seq_lengths = _load(session_dir / "seq_lengths.pth").long()
        obs_dir = session_dir / obs_src
        for i in range(len(seq_lengths)):
            src = obs_dir / f"episode_{i}.pth"
            dst = obs_out / f"episode_{global_idx}.pth"
            _link_or_copy(src, dst)
            global_idx += 1

    merged_meta = dict(metas[0])
    merged_meta["num_episodes"] = n_total
    merged_meta["merged_from"] = sessions
    merged_meta["obs_subdir"] = obs_dst
    if all("end_reason_counts" in m for m in metas):
        merged_counts = {
            "all_waypoints": 0,
            "stuck": 0,
            "max_steps": 0,
        }
        for meta in metas:
            for key, value in meta["end_reason_counts"].items():
                merged_counts[key] = merged_counts.get(key, 0) + int(value)
        merged_meta["end_reason_counts"] = merged_counts

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(padded_actions, out_dir / "actions.pth")
    torch.save(padded_states, out_dir / "states.pth")
    torch.save(padded_costs, out_dir / "costs.pth")
    torch.save(torch.tensor(flat_seq_lengths, dtype=torch.long), out_dir / "seq_lengths.pth")
    torch.save(merged_meta, out_dir / "meta.pth")

    print(f"[DONE] Merged {n_total} episodes into {out_dir}")
    print(f"       actions: {tuple(padded_actions.shape)}")
    print(f"       states:  {tuple(padded_states.shape)}")
    print(f"       obses:   {obs_out} ({global_idx} files)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge humanoid G1 dataset sessions.")
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/storage1/fs1/sibai/Active/ihab/research_new/datasets_dino/humanoid_g1",
        help="Parent directory containing session1, session2, ...",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Merged dataset output directory (default: <base-dir>/session_dir)",
    )
    parser.add_argument(
        "--sessions",
        nargs="+",
        default=["session1", "session2", "session3", "session4"],
    )
    args = parser.parse_args()
    merge_sessions(
        Path(args.base_dir),
        args.sessions,
        Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
