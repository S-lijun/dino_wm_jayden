#!/usr/bin/env bash
# CLS + concat-action variant of the start-goal path SAC pipeline.
# Does NOT replace scripts/run_train_HJ_humanoid_sac_path.sh
# (that one keeps patch tokens + late fusion).
#
# Same layout / collect / h_s as the path pipeline, but:
#   z = DINOv2 CLS (384) [+ proprio], no look-ahead
#   Q = MLP(cat(z, a))  — 3-D action concat at the first Linear
# Same frozen WM (wm_ckpt_18-27-17); only the extracted token changes.
#
# Usage:
#   bash scripts/run_train_HJ_humanoid_sac_path_cls.sh
#   bash scripts/run_train_HJ_humanoid_sac_path_cls.sh --action_reg_coef 0.5
#   bash scripts/run_train_HJ_humanoid_sac_path_cls.sh \
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
echo "[INFO] SAC path CLS+concat train -> runs/sac_hj_humanoid_path/"
echo "[INFO] z=DINOv2 CLS (384); Q=MLP(cat(z, a)); no late fusion"
echo "[INFO] layout: start + 2 perp vias (±2.5m) + goal; arena x[-1,6] y[-4,2]"
echo "[INFO] obstacles: 10% none; else 50/50 one vs two bins on start-goal perp ±2.5m (independent of vias)"
echo "[INFO] collect: alternate waypoint / Q-gate switch each episode"
echo "[INFO] buffer warmup 4000 waypoint steps then 4000 critic updates; buffer-size unchanged"
echo "[INFO] λ_good=0 (no a_good MSE); pass --action_reg_coef >0 to enable"
echo "[INFO] l/h_s: lidar d_min-1.5 in ±60° (120°) cone; outside cone l=2; non-foot contact (ignore ankle_roll soles)"

exec "${ISAAC_PYTHON}" train_HJ_humanoid_sac_path.py \
  --headless \
  --visual_mode rtx_rgb \
  --dino_ckpt_dir "${WM_CKPT_DIR}" \
  --dino_encoder "${WM_ENCODER}" \
  --visual_feature cls \
  --critic_fusion concat \
  --with_proprio \
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
