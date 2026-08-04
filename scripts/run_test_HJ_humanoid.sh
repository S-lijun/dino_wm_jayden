#!/usr/bin/env bash
# Closed-loop test: waypoint nominal + HJ switching (critic-only gate).
#
# Usage:
#   bash scripts/run_test_HJ_humanoid.sh \
#     --policy_path runs/ddpg_hj_humanoid/.../epoch_id_10/policy.pth
#
# Optional env:
#   ISAAC_PYTHON   python binary (default: python)
#   WM_CKPT_DIR    parent of encoder folder (default: C:/ on Windows-style paths)
#   WM_ENCODER     encoder folder (default: wm_ckpt_18-27-17)
#   MODE           switching | waypoint_only | safe_only (default: switching)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-python}"

WM_CKPT_DIR="${WM_CKPT_DIR:-C:/}"
WM_ENCODER="${WM_ENCODER:-wm_ckpt_18-27-17}"
MODE="${MODE:-switching}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
mkdir -p "${MPLCONFIGDIR}" "${REPO_ROOT}/humanoid_test"

cd "${REPO_ROOT}"

exec "${ISAAC_PYTHON}" test_HJ_humanoid.py \
  --headless \
  --visual_mode rtx_rgb \
  --dino_ckpt_dir "${WM_CKPT_DIR}" \
  --dino_encoder "${WM_ENCODER}" \
  --with_proprio \
  --config train_HJ_configs.yaml \
  --device cuda:0 \
  --mode "${MODE}" \
  --num_runs 5 \
  --safety_threshold 0.0 \
  --max_visual_steps 400 \
  "$@"
