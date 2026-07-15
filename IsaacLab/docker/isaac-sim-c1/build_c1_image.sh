#!/usr/bin/env bash
# Build and push C1-compatible Isaac Sim image on Compute1 via LSF docker_build.
#
# Prerequisites (one-time):
#   1. Docker Hub account
#   2. NGC API key (https://org.ngc.nvidia.com/setup/api-key) to pull the base image
#
# One-time logins on compute1-client-1:
#   # Docker Hub
#   LSB_DOCKER_LOGIN_ONLY=1 bsub -G compute-sibai -q general-interactive -Is -a 'docker_build' -- .
#   # NGC (username: $oauthtoken, password: <NGC API key>)
#   LSB_DOCKER_LOGIN_ONLY=1 LSB_DOCKER_LOGIN_SERVER=nvcr.io \
#     bsub -G compute-sibai -q general-interactive -Is -a 'docker_build' -- .
#
# Usage:
#   cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden/IsaacLab/docker/isaac-sim-c1
#   ./build_c1_image.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/.env.c1"

FULL_IMAGE="${DOCKERHUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}"
ISAACLAB_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "[INFO] Building ${FULL_IMAGE}"
echo "[INFO] Base image: nvcr.io/nvidia/isaac-sim:5.1.0 (official, unmodified content + chmod only)"
echo "[INFO] Build context: ${SCRIPT_DIR}"
echo "[INFO] This may take a long time (large image). Monitor with: bjobs -u \$USER"

cd "${ISAACLAB_DIR}"

bsub -G "${LSF_GROUP}" -q "${LSF_QUEUE}" -Is \
  -n 4 -R 'rusage[mem=32GB]' -M 30GB \
  -o "${SCRIPT_DIR}/build_%J.out" -e "${SCRIPT_DIR}/build_%J.err" \
  -a "docker_build(${FULL_IMAGE})" -- "${SCRIPT_DIR}"

echo "[INFO] Monitor progress (docker writes to stderr):"
echo "       tail -f ${SCRIPT_DIR}/build_<JOBID>.err"
echo "       Then run: ./run_c1_isaac.sh"
