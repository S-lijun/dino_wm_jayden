#!/usr/bin/env bash
# Train critic-only HJ for QP safety filter on Isaac G1 + DINO-WM.
#
# Usage:
#   bash scripts/run_train_HJ_humanoid_qp.sh
#   bash scripts/run_train_HJ_humanoid_qp.sh --resume_policy runs/qp_hj_humanoid/.../epoch_id_5/policy.pth
#
# Optional env:
#   WM_CKPT_DIR   parent of encoder folder (default: /workspace)
#   WM_ENCODER    encoder / run folder name (default: wm_ckpt_18-27-17)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"

WM_CKPT_DIR="${WM_CKPT_DIR:-/workspace}"
WM_ENCODER="${WM_ENCODER:-wm_ckpt_18-27-17}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ ! -f "${WM_CKPT_DIR}/${WM_ENCODER}/hydra.yaml" ]]; then
  echo "[ERROR] Missing ${WM_CKPT_DIR}/${WM_ENCODER}/hydra.yaml"
  echo "  Set WM_CKPT_DIR / WM_ENCODER, or put the WM run under /workspace/wm_ckpt_18-27-17"
  exit 1
fi

mkdir -p "${MPLCONFIGDIR}" "${REPO_ROOT}/runs"

cd "${REPO_ROOT}"

# Stale Isaac/train processes leave the GPU wedged → first collect step hangs forever.
if pgrep -f 'train_HJ_humanoid_qp.py' >/dev/null 2>&1; then
  echo "[ERROR] Another train_HJ_humanoid_qp.py is already running."
  echo "  Kill it first: pkill -9 -f train_HJ_humanoid_qp.py"
  pgrep -af 'train_HJ_humanoid_qp.py' || true
  exit 1
fi
GPU_MEM="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')"
if [[ -n "${GPU_MEM}" && "${GPU_MEM}" -gt 500 ]]; then
  echo "[WARN] GPU memory already ${GPU_MEM} MiB — leftover process? Check: nvidia-smi"
fi

echo "[INFO] WM: ${WM_CKPT_DIR}/${WM_ENCODER}"
echo "[INFO] python: ${ISAAC_PYTHON}"
echo "[INFO] QP critic-only train (yaw free) -> runs/qp_hj_humanoid/"

exec "${ISAAC_PYTHON}" train_HJ_humanoid_qp.py \
  --headless \
  --visual_mode rtx_rgb \
  --dino_ckpt_dir "${WM_CKPT_DIR}" \
  --dino_encoder "${WM_ENCODER}" \
  --with_proprio \
  --config train_HJ_configs.yaml \
  --device cuda:0 \
  --training-num 1 \
  --test-num 1 \
  --gamma-pyhj 0.98 \
  --critic_warmup_updates 1000 \
  --critic_action_samples 64 \
  --train_collect_noise 0.1 \
  --y_bound 0 \
  --wandb_video_every 5 \
  --buffer-size 10000 \
  "$@"
