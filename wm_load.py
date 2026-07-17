"""Lightweight DINO-WM checkpoint loader.

Safe to import under Isaac Sim: does not pull in ``env.venv`` / project ``utils``,
which collide with OpenCV's ``cv2.utils`` on Isaac's ``sys.path``.
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
import torch

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_ckpt(snapshot_path, device):
    """Load full WM pickle. Must use weights_only=False (contains ViTPredictor etc.)."""
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


# Alias kept for callers that used the old name.
load_ckpt_not_weights_only = load_ckpt


def _build_model(result, train_cfg, num_action_repeat, device):
    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(train_cfg.encoder)
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(_REPO_ROOT, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path, weights_only=False)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = ckpt
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint "
                "and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if Path(model_ckpt).exists():
        model_ckpt = Path(model_ckpt)
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")
    return _build_model(result, train_cfg, num_action_repeat, device)


load_model_not_weights_only = load_model
