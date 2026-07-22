#!/usr/bin/env bash
# Train DDPG HJ safety filter on Isaac G1 + DINO-WM latent space.
#
# Usage:
#   bash scripts/run_train_HJ_humanoid.sh
#   # resume from epoch_id_5, continue until total-episodes (default 120 in YAML):
#   bash scripts/run_train_HJ_humanoid.sh \
#     --resume_policy runs/ddpg_hj_humanoid/.../epoch_id_5/policy.pth
#
# Optional env:
#   WM_CKPT_DIR   parent of encoder folder (default: /workspace)
#   WM_ENCODER    encoder / run folder name (default: wm_ckpt_18-27-17)
#   WANDB_MODE    default: disabled

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"

WM_CKPT_DIR="${WM_CKPT_DIR:-/workspace}"
WM_ENCODER="${WM_ENCODER:-wm_ckpt_18-27-17}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${MPLCONFIGDIR}" "${REPO_ROOT}/runs"

cd "${REPO_ROOT}"

echo "[INFO] WM: ${WM_CKPT_DIR}/${WM_ENCODER}"
echo "[INFO] python: ${ISAAC_PYTHON}"

exec "${ISAAC_PYTHON}" train_HJ_humanoid.py \
  --headless \
  --visual_mode depth_rgb \
  --dino_ckpt_dir "${WM_CKPT_DIR}" \
  --dino_encoder "${WM_ENCODER}" \
  --with_proprio \
  --config train_HJ_configs.yaml \
  --device cuda:0 \
  --training-num 1 \
  --test-num 1 \
  "$@"
