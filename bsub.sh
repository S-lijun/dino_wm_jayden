#!/bin/bash
#BSUB -o /storage1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs_yuxuan/newnew/output_%J.log
#BSUB -e /storage1/sibai/Active/ihab/research_new/dino_wm/scratch_ihab_files/logs_yuxuan/newnew/error_%J.log
#BSUB -R "rusage[mem=40]"
#BSUB -gpu "num=1:mode=shared:gmodel=NVIDIAA100_SXM4_80GB"
#BSUB -J PythonGPUJob
export PATH="/storage1/sibai/Active/yuxuan/anaconda3/bin:$PATH"
source activate 
conda activate dino

export WANDB_API_KEY=7893bf6676aaa0213e6da2edbc8f4b42fa816084
wandb login
cd /storage1/sibai/Active/ihab/research_new/dino_wm
python /storage1/sibai/Active/ihab/research_new/dino_wm/train_failure_classifier_temp.py --seed 1 --task maniskill3000classif --epochs 2
