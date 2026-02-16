#!/bin/bash
#SBATCH --job-name=MOSES_h1024_L6
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=240:00:00
#SBATCH --output=moses_h1024_L6_%j.out
#SBATCH --error=moses_h1024_L6_%j.err

source molenv/bin/activate
cd /sci/labs/orzuk/shaulytolk/GRASSY-Net

export PYTHONPATH=/sci/labs/orzuk/shaulytolk/GRASSY-Net:$PYTHONPATH
export XDG_CACHE_HOME=/tmp/cache_$SLURM_JOB_ID
export PIP_NO_CACHE_DIR=1
export WANDB_DIR=/sci/labs/orzuk/shaulytolk/GRASSY-Net/wandb
export WANDB_CACHE_DIR=/tmp/wandb_cache_$SLURM_JOB_ID

python -m grassy_dit.train --config configs/MOSES/moses_dit_h1024_L6.yaml
