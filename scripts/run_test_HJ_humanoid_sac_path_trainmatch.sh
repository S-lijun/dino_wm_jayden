#!/usr/bin/env bash
# Eval that matches the 0902 start_goal train command.
# Does NOT replace scripts/run_test_HJ_humanoid_sac_path.sh or _cls.sh
# (those use start_goal_perp, no a_nom noise, |v|=0.5).
#
# Usage:
#   bash scripts/run_test_HJ_humanoid_sac_path_trainmatch.sh \
#     runs/sac_hj_humanoid_path/.../epoch_id_N/policy.pth
#   bash scripts/run_test_HJ_humanoid_sac_path_trainmatch.sh <policy.pth> --num_runs 3
#   bash scripts/run_test_HJ_humanoid_sac_path_trainmatch.sh <policy.pth> --mode switching

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
echo "[INFO] train-match: start_goal, band ±1 m, |v|~U[0.05,0.95], N(0,σ) σ∈{0.1,0.2,0.3} vy=yaw, z_fuse=5"

exec "${ISAAC_PYTHON}" test_HJ_humanoid_sac_path_trainmatch.py \
  --headless \
  --visual_mode rtx_rgb \
  --dino_ckpt_dir "${WM_CKPT_DIR}" \
  --dino_encoder "${WM_ENCODER}" \
  --visual_feature cls \
  --critic_fusion concat \
  --with_proprio \
  --config train_HJ_configs.yaml \
  --device cuda:0 \
  --policy_path "${POLICY_PATH}" \
  --mode compare \
  --num_runs 5 \
  --goal_radius 0.1 \
  --max_visual_steps 800 \
  --max_episode_steps 20000 \
  --gamma-pyhj 0.98 \
  --y_bound 0 \
  --use_arena_bounds \
  --arena_x_min -1 \
  --arena_x_max 6 \
  --arena_y_min -4 \
  --arena_y_max 2 \
  --skip_arena_oob_from_buffer \
  --lidar_distance_threshold 1.0 \
  --lidar_h_half_fov_deg 60 \
  --include_contact_in_hs \
  --contact_hs -2.5 \
  --randomize_nom_speed \
  --nom_speed_min 0.05 \
  --nom_speed_max 0.95 \
  --nom_act_noise \
  --nom_act_noise_dist gaussian \
  --nom_act_noise_stds 0.1,0.2,0.3 \
  --yaw_limit 1.0 \
  --waypoint_layout start_goal \
  --perp_offset 1.0 \
  --min_start_goal_dist 5.0 \
  --max_n_obstacles 2 \
  --path_obstacle_layout \
  --obstacle_absent_prob 0.1 \
  --two_obstacle_prob 0.5 \
  --z_fuse_frames 5 \
  "$@"
