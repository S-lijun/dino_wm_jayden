#!/usr/bin/env bash
# Test SAC HJ safety filter on the start–goal path layout.
#
# Each trial freezes the same scene:
#   start + 2 perpendicular vias + goal
#   0 or 1 bin (never two); when present, the 3rd waypoint (trans2) is inside
#   the bin's danger disk (r=1.5)
# then runs waypoint_only / SF-only / switching, writes 3 videos + 1 overlay PNG.
#
# Usage:
#   bash scripts/run_test_HJ_humanoid_sac_path.sh \
#     runs/sac_hj_humanoid_path/.../epoch_id_N/policy.pth
#   bash scripts/run_test_HJ_humanoid_sac_path.sh <policy.pth> --num_runs 3
#   bash scripts/run_test_HJ_humanoid_sac_path.sh <policy.pth> --mode switching
#   bash scripts/run_test_HJ_humanoid_sac_path.sh <policy.pth> --easy

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

# VNC DISPLAY=:1 + VirtualGL makes Isaac RTX crash in createHydraEngine.
if [[ -n "${DISPLAY:-}" || -n "${VGL_DISPLAY:-}" ]]; then
  echo "[INFO] Unsetting DISPLAY=${DISPLAY-} VGL_DISPLAY=${VGL_DISPLAY-} for headless RTX"
  unset DISPLAY VGL_DISPLAY VNCGL_DISPLAY || true
fi

if [[ ! -f "${WM_CKPT_DIR}/${WM_ENCODER}/hydra.yaml" ]]; then
  echo "[ERROR] Missing ${WM_CKPT_DIR}/${WM_ENCODER}/hydra.yaml"
  exit 1
fi

mkdir -p "${MPLCONFIGDIR}"
cd "${REPO_ROOT}"

echo "[INFO] WM: ${WM_CKPT_DIR}/${WM_ENCODER}"
echo "[INFO] policy: ${POLICY_PATH}"
echo "[INFO] compare modes: waypoint_only / safe_only / switching"
echo "[INFO] layout: start + trans1 + trans2 + goal; 0 or 1 bin; wp3 in danger r=1.5"

exec "${ISAAC_PYTHON}" test_HJ_humanoid_sac_path.py \
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
  --no_bin_every 2 \
  --goal_radius 0.1 \
  --danger_radius 1.5 \
  --max_visual_steps 800 \
  --max_episode_steps 20000 \
  --y_bound 0 \
  --use_arena_bounds \
  --arena_x_min -1 \
  --arena_x_max 6 \
  --arena_y_min -4 \
  --arena_y_max 2 \
  --lidar_distance_threshold 1.5 \
  --lidar_h_half_fov_deg 60 \
  --include_contact_in_hs \
  --contact_hs -1.5 \
  --yaw_limit 1.0 \
  --waypoint_layout start_goal_perp \
  --perp_offset 2.5 \
  --min_start_goal_dist 4.0 \
  --max_n_obstacles 1 \
  --path_obstacle_layout \
  "$@"
