#!/usr/bin/env bash
# Test SAC HJ safety filter (default: compare SF-only / waypoint / switching).
#
# Each trial freezes the same scene:
#   start → front → left|right in danger disk (bin center, r=0.3) → back disk (5.5,0) r=1
# runs all three controllers, writes 3 videos + 1 overlay traj PNG.
# Every 5 trials, exactly one has no blue bin (waypoints otherwise unchanged).
#
# Usage:
#   bash scripts/run_test_HJ_humanoid_sac.sh runs/sac_hj_humanoid/.../epoch_id_N/policy.pth
#   bash scripts/run_test_HJ_humanoid_sac.sh <policy.pth> --num_runs 3
#   bash scripts/run_test_HJ_humanoid_sac.sh <policy.pth> --mode switching   # single mode

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
echo "[INFO] compare modes: safe_only / waypoint_only / switching"
echo "[INFO] waypoints: danger pass r=0.3 @ bin, back=(5.5,-2) r=1"

exec "${ISAAC_PYTHON}" test_HJ_humanoid_sac.py \
  --headless \
  --visual_mode rtx_rgb \
  --dino_ckpt_dir "${WM_CKPT_DIR}" \
  --dino_encoder "${WM_ENCODER}" \
  --with_proprio \
  --config train_HJ_configs.yaml \
  --device cuda:0 \
  --policy_path "${POLICY_PATH}" \
  --mode compare \
  --num_runs 5 \
  --pass_radius 0.3 \
  --back_center_x 5.5 \
  --back_center_y -2.0 \
  --back_radius 1.0 \
  --goal_radius 0.1 \
  "$@"
