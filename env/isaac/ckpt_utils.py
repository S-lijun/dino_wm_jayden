"""Keep only the latest N ``epoch_id_*`` checkpoint directories."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import torch

_EPOCH_DIR = re.compile(r"^epoch_id_(\d+)$")


def save_epoch_checkpoint(policy, log_path: Path, epoch: int, keep: int = 2) -> Path:
    """Save ``policy.pth`` under ``log_path/epoch_id_{epoch}``, then prune older dirs."""
    log_path = Path(log_path)
    ckpt_dir = log_path / f"epoch_id_{int(epoch)}"
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    torch.save(policy.state_dict(), ckpt_dir / "policy.pth")
    prune_old_epoch_checkpoints(log_path, keep=keep)
    return ckpt_dir


def prune_old_epoch_checkpoints(log_path: Path, keep: int = 2) -> None:
    """Delete ``epoch_id_*`` dirs older than the newest ``keep`` (by epoch id)."""
    log_path = Path(log_path)
    if keep < 0 or not log_path.is_dir():
        return
    found: list[tuple[int, Path]] = []
    for p in log_path.iterdir():
        if not p.is_dir():
            continue
        m = _EPOCH_DIR.fullmatch(p.name)
        if m:
            found.append((int(m.group(1)), p))
    found.sort(key=lambda t: t[0])
    drop = found if keep == 0 else found[:-keep]
    for epoch_id, p in drop:
        shutil.rmtree(p, ignore_errors=True)
        print(f"[INFO] Removed old checkpoint epoch_id_{epoch_id} (keep last {keep})")
