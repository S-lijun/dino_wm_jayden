#!/usr/bin/env bash
# Start-goal path SAC on cat(DINO CLS, SpUNet, proprio).
# Does NOT replace scripts/run_train_HJ_humanoid_sac_path_lidar.sh
# (that one uses SpUNet only).
#
# z_vision = DINOv2 CLS from checkpoints/model_latest.pth (384)
# z_lidar  = frozen ScanNet SpUNet (96)
# z        = mean of the last 10 frames of cat(z_vision, z_lidar, z_proprio)
# Q        = MLP(cat(z, (vx, vy, yaw_rate)))
# Ray pattern: mid360
#
# Layout: start → goal, |S-G| >= 5 m. Bins in the perp band
# U[0.25, 1.5] m. Arena x[-1, 6] y[-4, 2].
# Obstacles: 10% none; else 50/50 one vs two bins.
# h_s = d_min - 1.0; non-foot contact → -2.5.
# |v| ~ U[0.05, 0.95] once per episode.
# a_nom (vx, vy, yaw) += N(0, σ), σ ~ {0.1, 0.2} once per episode.
# λ_good = 0.
#
# Usage:
#   bash scripts/run_train_HJ_humanoid_sac_path_vision_lidar.sh
#   bash scripts/run_train_HJ_humanoid_sac_path_vision_lidar.sh \
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

CONDA_ROOT="${CONDA_ROOT:-/home/humanoid-pc/miniconda3}"
CONDA_ENV="${CONDA_ENV:-g1_loco_sf}"
ISAAC_PYTHON="${ISAAC_PYTHON:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"
# Patched PyHJ lives in the G1_Loco_SF tree, not as a pip install.
PYHJ_PARENT="${PYHJ_PARENT:-/home/humanoid-pc/G1_Loco_SF}"

WM_CKPT_DIR="${WM_CKPT_DIR:-${REPO_ROOT}}"
WM_ENCODER="${WM_ENCODER:-wm_ckpt_18-27-17}"
SPUNET_CKPT="${SPUNET_CKPT:-${WM_CKPT_DIR}/${WM_ENCODER}/spunet_v1m1_scannet.pth}"
DINO_CKPT="${DINO_CKPT:-${WM_CKPT_DIR}/${WM_ENCODER}/checkpoints/model_latest.pth}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig}"
export PYTHONUNBUFFERED=1

if [[ ! -x "${ISAAC_PYTHON}" ]]; then
  echo "[ERROR] Missing Isaac python: ${ISAAC_PYTHON}"
  exit 1
fi

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if ! "${ISAAC_PYTHON}" -c "import PyHJ" >/dev/null 2>&1; then
  if [[ -d "${PYHJ_PARENT}/PyHJ" ]]; then
    export PYTHONPATH="${PYHJ_PARENT}:${PYTHONPATH}"
    echo "[INFO] PyHJ not installed; using ${PYHJ_PARENT}/PyHJ"
  else
    echo "[ERROR] Cannot import PyHJ, and ${PYHJ_PARENT}/PyHJ is missing."
    echo "        Install with scripts/setup_pyhj_isaac.sh or set PYHJ_PARENT."
    exit 1
  fi
fi

if [[ ! -f "${WM_CKPT_DIR}/${WM_ENCODER}/hydra.yaml" ]]; then
  echo "[ERROR] Missing ${WM_CKPT_DIR}/${WM_ENCODER}/hydra.yaml"
  exit 1
fi
if [[ ! -f "${SPUNET_CKPT}" ]]; then
  echo "[ERROR] Missing SpUNet checkpoint: ${SPUNET_CKPT}"
  exit 1
fi
if [[ ! -f "${DINO_CKPT}" ]]; then
  echo "[ERROR] Missing DINO checkpoint: ${DINO_CKPT}"
  exit 1
fi

mkdir -p "${MPLCONFIGDIR}" "${REPO_ROOT}/runs"
cd "${REPO_ROOT}"

echo "[INFO] python: ${ISAAC_PYTHON}"
echo "[INFO] WM: ${WM_CKPT_DIR}/${WM_ENCODER}"
echo "[INFO] DINO: ${DINO_CKPT}"
echo "[INFO] SpUNet: ${SPUNET_CKPT}"
echo "[INFO] SAC path vision+lidar concat train"
echo "[INFO] z=mean of last 10 cat(CLS 384, SpUNet 96, proprio); Q=MLP(cat(z, a))"
echo "[INFO] layout: start → goal (|S-G|>=5 m); bins perp U[0.25, 1.5] m"
echo "[INFO] arena x[-1,6] y[-4,2]"
echo "[INFO] obstacles: 10% none; else 50/50 one vs two bins"
echo "[INFO] collect: alternate waypoint / Q-gate switch each episode"
echo "[INFO] buffer warmup 4000 waypoint steps then 4000 critic updates"
echo "[INFO] λ_good=0"
echo "[INFO] l/h_s: lidar d_min-1.0; contact → -2.5"
echo "[INFO] nom |v| ~ U[0.05, 0.95] once per episode"
echo "[INFO] nom (vx,vy,yaw) += N(0, σ), σ~{0.1,0.2} once per episode; a_sf untouched"
echo "[INFO] gamma-pyhj=0.9"
echo "[INFO] tie_vy_yaw: one score s in (0,1) maps to both vy and yaw_rate"

exec "${ISAAC_PYTHON}" train_HJ_humanoid_sac_path_vision_lidar.py \
  --headless \
  --visual_mode rtx_rgb \
  --obs_mode wm_spunet \
  --visual_feature cls \
  --lidar_pattern mid360 \
  --dino_ckpt_dir "${WM_CKPT_DIR}" \
  --dino_encoder "${WM_ENCODER}" \
  --spunet_ckpt "${SPUNET_CKPT}" \
  --with_proprio \
  --critic_fusion concat \
  --config train_HJ_configs.yaml \
  --device cuda:0 \
  --training-num 1 \
  --test-num 1 \
  --gamma-pyhj 0.9 \
  --buffer_warmup_steps 4000 \
  --critic_warmup_updates 4000 \
  --actor_bc_warmup_updates 0 \
  --action_reg_coef 0.0 \
  --a_good_hold_steps 0 \
  --boundary_reg_coef 0.0 \
  --tie_vy_yaw \
  --y_bound 0 \
  --use_arena_bounds \
  --arena_x_min -1 \
  --arena_x_max 6 \
  --arena_y_min -4 \
  --arena_y_max 2 \
  --skip_arena_oob_from_buffer \
  --lidar_distance_threshold 1.0 \
  --include_contact_in_hs \
  --contact_hs -2.5 \
  --randomize_nom_speed \
  --nom_speed_min 0.05 \
  --nom_speed_max 0.95 \
  --nom_speed_mean 0.5 \
  --nom_speed_std 0.3 \
  --nom_act_noise \
  --nom_act_noise_dist gaussian \
  --nom_act_noise_stds 0.1,0.2 \
  --max_episode_steps 20000 \
  --yaw_limit 1.0 \
  --waypoint_layout start_goal \
  --perp_offset 1.5 \
  --min_start_goal_dist 5.0 \
  --max_n_obstacles 2 \
  --path_obstacle_layout \
  --obstacle_absent_prob 0.1 \
  --two_obstacle_prob 0.5 \
  --alternate_collect \
  --auto_alpha \
  --z_fuse_frames 10 \
  "$@"
