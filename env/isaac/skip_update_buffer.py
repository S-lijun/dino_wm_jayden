"""Replay buffer proxy that drops arena-OOB transitions from training.

Used only by the start-goal path SAC pipeline. The original Collector /
VectorReplayBuffer path is unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from PyHJ.data import Batch


def _as_1d(value) -> np.ndarray:
    return np.asarray(value).reshape(-1)


def batch_skip_update_mask(batch: Batch) -> np.ndarray:
    """True where this transition must not enter critic/actor updates."""
    info = getattr(batch, "info", None)
    if info is None:
        return np.zeros((len(batch),), dtype=bool)
    if hasattr(info, "skip_update"):
        return _as_1d(info.skip_update) > 0.5
    if hasattr(info, "end_reason"):
        return _as_1d(info.end_reason).astype(np.int32) == 4
    if isinstance(info, np.ndarray) and info.dtype == object:
        out = []
        for item in info:
            if not isinstance(item, dict):
                out.append(False)
                continue
            if float(item.get("skip_update", 0.0)) > 0.5:
                out.append(True)
            else:
                out.append(int(item.get("end_reason", 0)) == 4)
        return np.asarray(out, dtype=bool)
    return np.zeros((len(batch),), dtype=bool)


class SkipUpdateReplayBuffer:
    """Delegates to a VectorReplayBuffer, but does not store skip_update steps.

    The last in-bounds transition of that episode is marked done so n-step /
    episode boundaries stay valid. The wall-hit (s, a, s') itself is never
    sampled by ``learn``.
    """

    def __init__(self, buffer):
        self._buffer = buffer

    def add(self, batch: Batch, buffer_ids=None):
        skip = batch_skip_update_mask(batch)
        if skip.size == 0 or not bool(np.any(skip)):
            return self._buffer.add(batch, buffer_ids=buffer_ids)
        if bool(np.all(skip)):
            return self._seal_and_drop(batch, buffer_ids)
        # training_num is 1; mixed skip in one add is unexpected.
        keep = ~skip
        kept = batch[keep]
        kept_ids = None if buffer_ids is None else np.asarray(buffer_ids)[keep]
        ptrs, rews, lens, idxs = self._buffer.add(kept, buffer_ids=kept_ids)
        self._seal_and_drop(batch[skip], None if buffer_ids is None else np.asarray(buffer_ids)[skip])
        return ptrs, rews, lens, idxs

    def _seal_and_drop(self, batch: Batch, buffer_ids):
        buf = self._buffer
        if buffer_ids is None:
            n = 1 if not hasattr(buf, "buffer_num") else int(buf.buffer_num)
            buffer_ids = np.arange(n)
        else:
            buffer_ids = np.asarray(buffer_ids)
        n = int(len(buffer_ids))
        ep_rews = np.zeros(n, dtype=np.float64)
        ep_lens = np.zeros(n, dtype=np.int64)
        ep_idxs = np.zeros(n, dtype=np.int64)
        ptrs = np.zeros(n, dtype=np.int64)
        for i, bid in enumerate(buffer_ids):
            bid = int(bid)
            last = int(buf.last_index[bid])
            ptrs[i] = last
            if hasattr(buf, "buffers"):
                sub = buf.buffers[bid]
                ep_rews[i] = float(getattr(sub, "_ep_rew", 0.0))
                ep_lens[i] = int(getattr(sub, "_ep_len", 0))
                offset = int(buf._offset[bid]) if hasattr(buf, "_offset") else 0
                ep_idxs[i] = int(getattr(sub, "_ep_idx", 0)) + offset
                if len(buf) > 0:
                    buf.truncated[last] = True
                    buf.done[last] = True
                sub._ep_rew = 0.0
                sub._ep_len = 0
                sub._ep_idx = sub._index
            elif len(buf) > 0:
                buf.truncated[last] = True
                buf.done[last] = True
                ep_rews[i] = float(getattr(buf, "_ep_rew", 0.0))
                ep_lens[i] = int(getattr(buf, "_ep_len", 0))
                buf._ep_rew, buf._ep_len = 0.0, 0
                buf._ep_idx = buf._index
        return ptrs, ep_rews, ep_lens, ep_idxs

    def __len__(self):
        return len(self._buffer)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._buffer, name)

    def __getitem__(self, index):
        return self._buffer[index]
