#!/usr/bin/env bash
# Test SAC HJ safety filter (compare waypoint_only / SF-only / switching).
#
# Each trial freezes the same scene (start→front→left|right@r=0.2→back),
# runs all three controllers, writes 3 videos + 1 overlay traj PNG.
#
# Usage:
#   bash scripts/run_test_HJ_humanoid_sac.sh runs/sac_hj_humanoid/.../epoch_id_N/policy.pth
#   bash scripts/run_test_HJ_humanoid_sac.sh <policy.pth> --num_runs 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"

POLICY_PATH="${1:?Usage: $0 /path/to/epoch_id_N/policy.pth}"
shift || true

WM_CKPT_DIR="${WM_CKPT_DIR:-/workspace}"
WM_ENCODER="${WM_ENCODER:-wm_ckpt_18-27-17}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${MPLCONFIGDIR}"
cd "${REPO_ROOT}"

echo "[INFO] WM: ${WM_CKPT_DIR}/${WM_ENCODER}"
echo "[INFO] policy: ${POLICY_PATH}"
echo "[INFO] compare modes: waypoint_only / safe_only / switching"

exec "${ISAAC_PYTHON}" test_HJ_humanoid_sac.py \
  --headless \
  --visual_mode rtx_rgb \
  --dino_ckpt_dir "${WM_CKPT_DIR}" \
  --dino_encoder "${WM_ENCODER}" \
  --with_proprio \
  --config train_HJ_configs.yaml \
  --device cuda:0 \
  --policy_path "${POLICY_PATH}" \
  --num_runs 5 \
  "$@"
