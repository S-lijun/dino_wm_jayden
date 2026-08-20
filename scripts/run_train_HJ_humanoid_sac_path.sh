#!/usr/bin/env bash
# Train SAC HJ safety filter with start-goal path layout (does NOT replace
# scripts/run_train_HJ_humanoid_sac.sh).
#
# Each episode: start + 2 perpendicular transition waypoints + goal.
# Arena rectangle x in [-1, 6], y in [-4, 2]; edge hit resets and is not
# stored for updates. h_s = d_min - 1.5, plus non-foot obstacle contact
# (left/right ankle_roll soles ignored) caps l at contact_hs=-1.5.
# l uses a ±60° (120°) forward cone so labels match camera-visible
# obstacles; outside the cone l=2.0. Episode cap 20000 sim steps ≈ 100s
# (aisle default was 8000 ≈ 40s).
# Obstacles: 10% none; else 50/50 one vs two bins, placed ±2.5 m off the
# start-goal line independently of vias (vias may land in a bin's 1.5 m
# danger disk). Dual RayCasters → min range is l.
# Formal collect alternates waypoint-only and Q-gate switch per episode.
# λ_good=0.5 only when switch AND HJ<0; waypoint_good held 20 steps then refresh.
# 2D top-down traj PNG (obstacles + dashed r=1.5 failure-set circles).
#
# Usage:
#   bash scripts/run_train_HJ_humanoid_sac_path.sh
#   bash scripts/run_train_HJ_humanoid_sac_path.sh --action_reg_coef 0.5
#   bash scripts/run_train_HJ_humanoid_sac_path.sh --freeze_yaw
#   bash scripts/run_train_HJ_humanoid_sac_path.sh \
#     --resume_policy runs/sac_hj_humanoid_path/.../epoch_id_N/policy.pth

set -euo pipefail

# VNC DISPLAY=:1 + VirtualGL makes Isaac RTX crash in createHydraEngine.
# Headless training must not inherit that X server.
if [[ -n "${DISPLAY:-}" || -n "${VGL_DISPLAY:-}" ]]; then
  echo "[INFO] Unsetting DISPLAY=${DISPLAY-} VGL_DISPLAY=${VGL_DISPLAY-} for headless RTX"
  unset DISPLAY VGL_DISPLAY VNCGL_DISPLAY || true
fi

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
  exit 1
fi

mkdir -p "${MPLCONFIGDIR}" "${REPO_ROOT}/runs"
cd "${REPO_ROOT}"

echo "[INFO] WM: ${WM_CKPT_DIR}/${WM_ENCODER}"
echo "[INFO] SAC path train -> runs/sac_hj_humanoid_path/"
echo "[INFO] layout: start + 2 perp vias (±2.5m) + goal; arena x[-1,6] y[-4,2]"
echo "[INFO] obstacles: 10% none; else 50/50 one vs two bins on start-goal perp ±2.5m (independent of vias)"
echo "[INFO] collect: alternate waypoint / Q-gate switch each episode"
echo "[INFO] buffer warmup 4000 waypoint steps then 4000 critic updates; buffer-size unchanged"
echo "[INFO] λ_good=0.5 on switch+HJ<0; a_good held 20 steps then refresh"
echo "[INFO] l/h_s: lidar d_min-1.5 in ±60° (120°) cone; outside cone l=2; non-foot contact (ignore ankle_roll soles)"

exec "${ISAAC_PYTHON}" train_HJ_humanoid_sac_path.py \
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
  --buffer_warmup_steps 4000 \
  --critic_warmup_updates 4000 \
  --actor_bc_warmup_updates 0 \
  --action_reg_coef 0.5 \
  --a_good_hold_steps 20 \
  --boundary_reg_coef 0.0 \
  --y_bound 0 \
  --use_arena_bounds \
  --arena_x_min -1 \
  --arena_x_max 6 \
  --arena_y_min -4 \
  --arena_y_max 2 \
  --skip_arena_oob_from_buffer \
  --lidar_distance_threshold 1.5 \
  --lidar_h_half_fov_deg 60 \
  --include_contact_in_hs \
  --contact_hs -1.5 \
  --max_episode_steps 20000 \
  --yaw_limit 1.0 \
  --waypoint_layout start_goal_perp \
  --perp_offset 2.5 \
  --min_start_goal_dist 4.0 \
  --max_n_obstacles 2 \
  --path_obstacle_layout \
  --obstacle_absent_prob 0.1 \
  --two_obstacle_prob 0.5 \
  --alternate_collect \
  --auto_alpha \
  "$@"
