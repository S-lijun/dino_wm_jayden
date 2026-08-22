#!/usr/bin/env bash
# Train SAC HJ safety filter with joints + full lidar scan (path layout).
# Does NOT replace scripts/run_train_HJ_humanoid_sac_path.sh (DINO visual z).
#
# State: 12 joints and full lidar, each MLP-encoded to 256 then cat to 512 z.
# Critic late-fuses z=512 with action encoded to 512. Actor uses the same z.
# h_s: geometric XY distance to nearest bin minus 1.5 m (not lidar min).
# No DINO world model.
#
# Usage:
#   bash scripts/run_train_HJ_humanoid_sac_lidar.sh
#   bash scripts/run_train_HJ_humanoid_sac_lidar.sh --action_reg_coef 0.5
#   bash scripts/run_train_HJ_humanoid_sac_lidar.sh \
#     --resume_policy runs/sac_hj_humanoid_lidar/.../epoch_id_N/policy.pth

set -euo pipefail

# VNC DISPLAY=:1 + VirtualGL makes Isaac RTX crash in createHydraEngine.
if [[ -n "${DISPLAY:-}" || -n "${VGL_DISPLAY:-}" ]]; then
  echo "[INFO] Unsetting DISPLAY=${DISPLAY-} VGL_DISPLAY=${VGL_DISPLAY-} for headless RTX"
  unset DISPLAY VGL_DISPLAY VNCGL_DISPLAY || true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${MPLCONFIGDIR}" "${REPO_ROOT}/runs"
cd "${REPO_ROOT}"

echo "[INFO] SAC lidar+joints train -> runs/sac_hj_humanoid_lidar/"
echo "[INFO] state: 12-D legs + full lidar (45×azim ranges); h_s = geom XY − 1.5"
echo "[INFO] layout: start + 2 perp vias (±2.5m) + goal; arena x[-1,6] y[-4,2]"
echo "[INFO] collect: alternate waypoint / Q-gate switch each episode"
echo "[INFO] λ_good=0 (no a_good MSE); lidar/actor stream one layer 256"

exec "${ISAAC_PYTHON}" train_HJ_humanoid_sac_lidar.py \
  --headless \
  --visual_mode rtx_rgb \
  --obs_mode lidar_joint \
  --hs_mode geom \
  --critic-net 256 \
  --control-net 256 \
  --config train_HJ_configs.yaml \
  --device cuda:0 \
  --training-num 1 \
  --test-num 1 \
  --gamma-pyhj 0.98 \
  --buffer_warmup_steps 4000 \
  --critic_warmup_updates 4000 \
  --actor_bc_warmup_updates 0 \
  --action_reg_coef 0 \
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
