#!/usr/bin/env bash
# Switching eval: vision+lidar first, then lidar-only.
# Same seed-0 scenes. Both policies are the gamma-0.9, tie_vy_yaw epoch_id_120 runs.
#
# Leave it running:
#   cd /home/humanoid-pc/dino_wm_jayden
#   nohup bash scripts/run_test_HJ_humanoid_sac_path_vision_lidar_then_lidar.sh \
#     > test_vl_then_lidar.log 2>&1 &

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

VL_POLICY="${VL_POLICY:-runs/sac_hj_humanoid_path/sac-path-cls-spunet-dino-concat-wm_ckpt_18-27-17-1101_1347/epoch_id_120/policy.pth}"
LIDAR_POLICY="${LIDAR_POLICY:-runs/sac_hj_humanoid_path/sac-path-spunet-concat-wm_ckpt_18-27-17-1102_1604/epoch_id_120/policy.pth}"

if [[ ! -f "${VL_POLICY}" ]]; then
  echo "[ERROR] Missing vision+lidar policy: ${VL_POLICY}"
  exit 1
fi
if [[ ! -f "${LIDAR_POLICY}" ]]; then
  echo "[ERROR] Missing lidar-only policy: ${LIDAR_POLICY}"
  exit 1
fi

echo "[INFO] $(date '+%F %T') start vision+lidar"
echo "[INFO] policy: ${VL_POLICY}"
bash "${SCRIPT_DIR}/run_test_HJ_humanoid_sac_path_vision_lidar.sh" "${VL_POLICY}"
vl_status=$?
echo "[INFO] $(date '+%F %T') vision+lidar exit ${vl_status}"

echo "[INFO] $(date '+%F %T') start lidar-only"
echo "[INFO] policy: ${LIDAR_POLICY}"
bash "${SCRIPT_DIR}/run_test_HJ_humanoid_sac_path_lidar.sh" "${LIDAR_POLICY}"
lidar_status=$?
echo "[INFO] $(date '+%F %T') lidar-only exit ${lidar_status}"

echo "[INFO] done vision+lidar=${vl_status} lidar-only=${lidar_status}"
if [[ "${vl_status}" -ne 0 || "${lidar_status}" -ne 0 ]]; then
  exit 1
fi
