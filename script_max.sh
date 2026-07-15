#!/bin/bash
#BSUB -n 24
#BSUB -q general-interactive
#BSUB -R 'rusage[mem=182GB]'
#BSUB -M 180GB
#BSUB -R 'gpuhost'
#BSUB -gpu "num=1:gmodel=TeslaV100_SXM2_32GB"
#BSUB -a 'docker(continuumio/anaconda3:2021.11)'
#BSUB -W 24:00
#BSUB -J dino_wm_job
#BSUB -oo /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs_yuxuan/output%J.log
#BSUB -eo /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs_yuxuan/error%J.log
#BSUB -N
#BSUB -u y.yuxuan@wustl.edu
#BSUB -env "LSF_DOCKER_VOLUMES=/storage1/fs1/sibai/Active:/storage1/fs1/sibai/Active,LSF_DOCKER_SHM_SIZE=32g"

export CONDA_ENVS_DIRS="/storage1/fs1/sibai/Active/max/conda/envs"
export CONDA_PKGS_DIRS="/storage1/fs1/sibai/Active/max/conda/pkgs"
export PATH="/opt/conda/bin:$PATH"
export TORCH_HOME=/storage1/fs1/sibai/Active/ihab/tmp/torch
export WANDB_API_KEY=7893bf6676aaa0213e6da2edbc8f4b42fa816084
# export WANDB_DISABLE_GIT=True
source /opt/conda/etc/profile.d/conda.sh
conda activate dino_wm_ris
cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm
pip install --upgrade wandb
wandb login
# git config --global --add safe.directory /storage1/fs1/sibai/Active/ihab/research_new/dino_wm
#python train_HJ_mani_ft --dino_encoder full_scratch --use_latent_h
python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task dubins1800_continuous_cost  --without_proprio --epochs 2
python /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier.py --seed 1 --task dubins1800_continuous_cost  --without_proprio --finetune --epochs 2