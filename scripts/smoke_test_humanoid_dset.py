#!/usr/bin/env python3
"""Smoke test: 480:640 letterbox, 1:1 visual/state/action alignment."""

import os
import sys
import torch

os.environ.setdefault(
    "DATASET_DIR",
    "/storage1/fs1/sibai/Active/ihab/research_new/datasets_dino",
)

from datasets.humanoid_dset import HumanoidDataset, load_humanoid_slice_train_val
from datasets.img_transforms import letterbox_transform


def main():
    img_h = int(sys.argv[1]) if len(sys.argv) > 1 else 144
    img_w = int(sys.argv[2]) if len(sys.argv) > 2 else 192
    data_path = os.path.join(os.environ["DATASET_DIR"], "humanoid_g1", "session_dir")
    transform = letterbox_transform(img_h=img_h, img_w=img_w)
    dset = HumanoidDataset(
        data_path=data_path,
        transform=transform,
        normalize_action=False,
        normalize_states=False,
    )
    obs, act, state, info = dset.get_frames(0, [0, 1, 2, 3])
    h, w = obs["visual"].shape[-2], obs["visual"].shape[-1]
    print("visual shape:", obs["visual"].shape, f"expected HxW={img_h}x{img_w}")
    assert h == img_h and w == img_w, f"got {h}x{w}"
    assert act.shape[0] == 4 and act.shape[1] == dset.action_dim

    ds, _ = load_humanoid_slice_train_val(
        transform=transform,
        data_path=data_path,
        n_rollout=10,
        normalize_action=False,
        num_hist=3,
        num_pred=1,
        frameskip=1,
    )
    obs, act, state = ds["train"][0]
    assert obs["visual"].shape[1:] == (3, img_h, img_w)
    print("train slices (10 ep):", len(ds["train"]))
    print("OK")


if __name__ == "__main__":
    main()
