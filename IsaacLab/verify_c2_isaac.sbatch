#!/bin/bash
#SBATCH -J isaac_c2_verify
#SBATCH -A compute2-sibai
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o /storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden/IsaacLab/logs/verify_c2_%j.out
#SBATCH -e /storage1/fs1/sibai/Active/ihab/research_new/dino_wm_jayden/IsaacLab/logs/verify_c2_%j.err

# Submit on c2-login:
#   ml load ris && ml load slurm
#   sbatch --container-image='nvcr.io#nvidia/isaac-sim:5.1.0' \
#     --container-mounts='/storage1/fs1/sibai/Active/ihab/research_new:/workspace' \
#     verify_c2_isaac.sbatch

bash /workspace/dino_wm_jayden/IsaacLab/verify_c2_isaac.sh
