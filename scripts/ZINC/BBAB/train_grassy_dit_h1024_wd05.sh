#!/bin/bash
#SBATCH --job-name=BBAB_wd05
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=168:00:00
#SBATCH --output=BBAB_wd05_%j.out
#SBATCH --error=BBAB_wd05_%j.err

source molenv/bin/activate
cd /sci/labs/orzuk/shaulytolk/GRASSY-Net
export XDG_CACHE_HOME=/tmp/cache_$SLURM_JOB_ID
export PIP_NO_CACHE_DIR=1
export WANDB_DIR=/sci/labs/orzuk/shaulytolk/GRASSY-Net/wandb
export WANDB_CACHE_DIR=/tmp/wandb_cache_$SLURM_JOB_ID

python -m grassy_dit.train --config configs/ZINC/BBAB/BBAB_dit_h1024_wd05.yaml
