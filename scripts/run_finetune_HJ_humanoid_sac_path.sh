#!/usr/bin/env bash
# Offline real-robot fine-tune of the path SAC safety filter.
# No Isaac / no collect. Defaults: critics only, match the cls+concat sim SF.
#
# Usage:
#   bash scripts/run_finetune_HJ_humanoid_sac_path.sh \
#     --data_dir /path/to/real_data \
#     --resume_policy /workspace/runs/sac_hj_humanoid_path/.../epoch_id_N/policy.pth
#
#   bash scripts/run_finetune_HJ_humanoid_sac_path.sh \
#     --data_dir /path/to/real_data \
#     --resume_policy .../policy.pth \
#     --train_actor --actor_reg_coef 0.2
#
# Env overrides:
#   PYTHON, DINO_CKPT_DIR, DATA_DIR, RESUME_POLICY

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python}"
DINO_CKPT_DIR="${DINO_CKPT_DIR:-/workspace/wm_ckpt_18-27-17}"
DATA_DIR="${DATA_DIR:-}"
RESUME_POLICY="${RESUME_POLICY:-}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ ! -f "${DINO_CKPT_DIR}/hydra.yaml" ]]; then
  echo "[ERROR] Missing ${DINO_CKPT_DIR}/hydra.yaml"
  echo "  Set DINO_CKPT_DIR to the WM run folder (contains hydra.yaml and checkpoints/)."
  exit 1
fi

EXTRA_ARGS=()
if [[ -n "${DATA_DIR}" ]]; then
  EXTRA_ARGS+=(--data_dir "${DATA_DIR}")
fi
if [[ -n "${RESUME_POLICY}" ]]; then
  EXTRA_ARGS+=(--resume_policy "${RESUME_POLICY}")
fi

mkdir -p "${MPLCONFIGDIR}" "${REPO_ROOT}/runs"
cd "${REPO_ROOT}"

echo "[INFO] offline FT (critics only unless --train_actor)"
echo "[INFO] WM: ${DINO_CKPT_DIR}"
echo "[INFO] python: ${PYTHON}"
echo "[INFO] z: cls + proprio, fuse 10, Q=concat, h_s=dist-1.0"
echo "[INFO] trainer match sim: collect 8, epoch 2000 env-steps, update-per-step 0.125 (=250 updates), batch 512"

exec "${PYTHON}" finetune_HJ_humanoid_sac_path.py \
  --dino_ckpt_dir "${DINO_CKPT_DIR}" \
  --config train_HJ_configs.yaml \
  --device cuda:0 \
  --visual_feature cls \
  --critic_fusion concat \
  --z_fuse_frames 10 \
  --lidar_distance_threshold 1.0 \
  --yaw_limit 1.0 \
  --gamma-pyhj 0.98 \
  --step-per-collect 8 \
  --step-per-epoch 2000 \
  --update-per-step 0.125 \
  --batch_size-pyhj 512 \
  --total_epochs 50 \
  "${EXTRA_ARGS[@]}" \
  "$@"
