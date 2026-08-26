#!/usr/bin/env bash
# CLS + concat-action + bin-side-disk layout.
# Does NOT replace scripts/run_train_HJ_humanoid_sac_path.sh
# or scripts/run_train_HJ_humanoid_sac_path_cls.sh.
#
# z = DINOv2 CLS (384) [+ proprio]; Q = MLP(cat(z, a)).
# Same frozen WM (wm_ckpt_18-27-17).
#
# Scene (90%): start/goal centers |S-G|>=4 m; one bin ±1 m perp off that
# segment; two r=1 disks whose edges touch the bin center; pick one 50/50
# for the via; actual start/goal sampled in r=1 disks around the centers.
# Scene (10%): no bin, 6 random arena waypoints.
# Collect: alternate waypoint-only / Q-gate switch.
# |v| ~ U[0.1, 0.9] per episode; a_nom += N(0,1) each step; a_sf untouched.
#
# Usage:
#   bash scripts/run_train_HJ_humanoid_sac_path_cls_disk.sh
#   bash scripts/run_train_HJ_humanoid_sac_path_cls_disk.sh --action_reg_coef 0.5
#   bash scripts/run_train_HJ_humanoid_sac_path_cls_disk.sh \
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
echo "[INFO] SAC path CLS+concat bin-side-disk train -> runs/sac_hj_humanoid_path/"
echo "[INFO] z=DINOv2 CLS (384); Q=MLP(cat(z, a)); no late fusion"
echo "[INFO] layout 90%: start-disk(r=1) + 1 bin-side via-disk(r=1) + goal-disk(r=1)"
echo "[INFO]   bin ±1 m perp of start–goal centers (|S-G|>=4 m); via disk 50/50"
echo "[INFO] layout 10%: no bin, 6 random arena waypoints"
echo "[INFO] collect: alternate waypoint / Q-gate switch each episode"
echo "[INFO] buffer warmup 4000 waypoint steps then 4000 critic updates; buffer-size unchanged"
echo "[INFO] λ_good=0 (no a_good MSE); pass --action_reg_coef >0: HJ<0 any controller, freeze a_good until HJ>=0"
echo "[INFO] l/h_s: lidar d_min-1.5 in ±60° (120°) cone; outside cone l=2; +contact (non-foot → -2.5)"
echo "[INFO] nom |v| ~ U[0.1, 0.9] once per episode"
echo "[INFO] nom (vx,vy,yaw) += N(0,1) every step; a_sf untouched"

exec "${ISAAC_PYTHON}" train_HJ_humanoid_sac_path_cls_disk.py \
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
  --a_good_hold_steps 0 \
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
  --contact_hs -2.5 \
  --randomize_nom_speed \
  --nom_speed_min 0.1 \
  --nom_speed_max 0.9 \
  --nom_act_noise \
  --nom_act_noise_std 1.0 \
  --max_episode_steps 20000 \
  --yaw_limit 1.0 \
  --waypoint_layout bin_side_disk \
  --perp_offset 1.0 \
  --min_start_goal_dist 4.0 \
  --start_goal_region_r 1.0 \
  --via_region_r 1.0 \
  --n_free_waypoints 6 \
  --max_n_obstacles 1 \
  --path_obstacle_layout \
  --obstacle_absent_prob 0.1 \
  --alternate_collect \
  --auto_alpha \
  "$@"
