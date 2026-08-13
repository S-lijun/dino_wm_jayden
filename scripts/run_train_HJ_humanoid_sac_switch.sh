#!/usr/bin/env bash
# Train SAC HJ safety filter with Q-gate switching on Isaac G1 + DINO-WM.
#
# Formal collect (after waypoint buffer + critic warmup):
#   Q(z, a_nom) >= 0  → waypoint controller interacts with the env
#   Q(z, a_nom) <  0  → safety filter interacts with the env
# Actor is pure SAC (no λ_nom, no boundary). SF still updates every learn step;
# only the executed action is switched.
#
# Does NOT replace run_train_HJ_humanoid_sac.sh (that run stays SF-only collect
# with optional λ_nom / boundary).
#
# Usage:
#   bash scripts/run_train_HJ_humanoid_sac_switch.sh
#   bash scripts/run_train_HJ_humanoid_sac_switch.sh --freeze_yaw
#   bash scripts/run_train_HJ_humanoid_sac_switch.sh --force_right_pass
#   bash scripts/run_train_HJ_humanoid_sac_switch.sh --spawn_hemisphere_pass
#   bash scripts/run_train_HJ_humanoid_sac_switch.sh \
#     --resume_policy runs/sac_hj_humanoid/.../epoch_id_N/policy.pth

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
echo "[INFO] SAC avoid train + Q-gate switch -> runs/sac_hj_humanoid/"
echo "[INFO] collect: waypoint if Q(a_nom)>=0 else SF; actor=pure SAC (no λ_nom/boundary)"

exec "${ISAAC_PYTHON}" train_HJ_humanoid_sac.py \
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
  --critic_warmup_updates 1000 \
  --actor_bc_warmup_updates 0 \
  --action_reg_coef 0.0 \
  --boundary_reg_coef 0.0 \
  --y_bound 3.0 \
  --y_center -2.0 \
  --x_bound_max 4.5 \
  --auto_alpha \
  --switch_collect \
  --switch_threshold 0.0 \
  "$@"
