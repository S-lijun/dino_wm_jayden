#!/usr/bin/env bash
# Switching eval for the lidar-only SpUNet path policy.
# z = mean of the last 10 cat(SpUNet 96, proprio 10)
#
#   start_goal, |S-G|>=5 m, bins perp U[0.25, 1.25] m
#   |v| ~ U[0.4, 0.6] once per episode, no a_nom noise
#   success fails if the 360° distance to a bin center drops below 0.5 m
#   mode=switching, 100 runs, seed=0 (same layouts as the other two tests)
#   actor is (vx, u); u maps to both vy and yaw_rate
#
# Usage:
#   bash scripts/run_test_HJ_humanoid_sac_path_lidar.sh \
#     runs/sac_hj_humanoid_path/.../epoch_id_N/policy.pth

set -euo pipefail

if [[ -n "${DISPLAY:-}" || -n "${VGL_DISPLAY:-}" ]]; then
  echo "[INFO] Unsetting DISPLAY=${DISPLAY-} VGL_DISPLAY=${VGL_DISPLAY-} for headless RTX"
  unset DISPLAY VGL_DISPLAY VNCGL_DISPLAY || true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ROOT="${CONDA_ROOT:-/home/humanoid-pc/miniconda3}"
CONDA_ENV="${CONDA_ENV:-g1_loco_sf}"
ISAAC_PYTHON="${ISAAC_PYTHON:-${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python}"
PYHJ_PARENT="${PYHJ_PARENT:-/home/humanoid-pc/G1_Loco_SF}"

POLICY_PATH="${1:?Usage: $0 /path/to/epoch_id_N/policy.pth}"
shift || true

WM_CKPT_DIR="${WM_CKPT_DIR:-${REPO_ROOT}}"
WM_ENCODER="${WM_ENCODER:-wm_ckpt_18-27-17}"
SPUNET_CKPT="${SPUNET_CKPT:-${WM_CKPT_DIR}/${WM_ENCODER}/spunet_v1m1_scannet.pth}"
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
    exit 1
  fi
fi

if [[ ! -f "${POLICY_PATH}" ]]; then
  echo "[ERROR] Missing policy: ${POLICY_PATH}"
  exit 1
fi
if [[ ! -f "${WM_CKPT_DIR}/${WM_ENCODER}/hydra.yaml" ]]; then
  echo "[ERROR] Missing ${WM_CKPT_DIR}/${WM_ENCODER}/hydra.yaml"
  exit 1
fi
if [[ ! -f "${SPUNET_CKPT}" ]]; then
  echo "[ERROR] Missing SpUNet checkpoint: ${SPUNET_CKPT}"
  exit 1
fi

mkdir -p "${MPLCONFIGDIR}"
cd "${REPO_ROOT}"

echo "[INFO] python: ${ISAAC_PYTHON}"
echo "[INFO] SpUNet: ${SPUNET_CKPT}"
echo "[INFO] policy: ${POLICY_PATH}"
echo "[INFO] switching x100 seed=0: z=cat(SpUNet,proprio) fuse=10, bins U[0.25,1.25], |v|~U[0.4,0.6], no a_nom noise, fail_r=0.5, tie_vy_yaw"

exec "${ISAAC_PYTHON}" test_HJ_humanoid_sac_path_lidar.py \
  --headless \
  --visual_mode rtx_rgb \
  --device cuda:0 \
  --obs_mode lidar_spunet \
  --lidar_pattern mid360 \
  --dino_ckpt_dir "${WM_CKPT_DIR}" \
  --dino_encoder "${WM_ENCODER}" \
  --spunet_ckpt "${SPUNET_CKPT}" \
  --critic_fusion concat \
  --with_proprio \
  --z_fuse_frames 10 \
  --tie_vy_yaw \
  --policy_path "${POLICY_PATH}" \
  --waypoint_layout start_goal \
  --perp_offset 1.25 \
  --perp_keepout 0.25 \
  --min_start_goal_dist 5.0 \
  --max_n_obstacles 2 \
  --obstacle_absent_prob 0.1 \
  --two_obstacle_prob 0.5 \
  --lidar_distance_threshold 1.0 \
  --include_contact_in_hs \
  --contact_hs -2.5 \
  --yaw_limit 1.0 \
  --randomize_nom_speed \
  --nom_speed_min 0.4 \
  --nom_speed_max 0.6 \
  --no_nom_act_noise \
  --success_fail_radius 0.5 \
  --mode switching \
  --num_runs 100 \
  --seed 0 \
  "$@"
