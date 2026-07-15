#!/bin/bash
#BSUB -n 24
#BSUB -q general
#BSUB -G compute-sibai
#BSUB -R 'rusage[mem=102GB]'
#BSUB -M 100GB
#BSUB -R 'gpuhost'
#BSUB -gpu "num=1"
#BSUB -a 'docker(maniskill/base)'
#BSUB -W 600:00
#BSUB -J dino_wm_job
#BSUB -oo /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs_yuxuan/newnew/output%J.log
#BSUB -eo /storage1/fs1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs_yuxuan/newnew/error%J.log
#BSUB -N
#BSUB -u y.yuxuan@wustl.edu
#BSUB -env "LSF_DOCKER_VOLUMES=/storage1/fs1/sibai/Active:/storage1/fs1/sibai/Active,LSF_DOCKER_SHM_SIZE=32g"


export CONDA_ENVS_DIRS="/storage1/fs1/sibai/Active/ihab/conda/envs"
export CONDA_PKGS_DIRS="/storage1/fs1/sibai/Active/ihab/conda/pkgs"
export PATH="/opt/conda/bin:$PATH"
export TORCH_HOME=/storage1/fs1/sibai/Active/ihab/tmp/torch
export WANDB_API_KEY=7893bf6676aaa0213e6da2edbc8f4b42fa816084

source /opt/conda/etc/profile.d/conda.sh
conda activate mani
cd /storage1/fs1/sibai/Active/ihab/research_new/dino_wm
wandb login
 
