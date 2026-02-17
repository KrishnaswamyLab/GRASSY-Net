#!/bin/bash
#SBATCH --job-name=qm9_moment
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=qm9_moment_%j.out
#SBATCH --error=qm9_moment_%j.err

source molenv/bin/activate
cd /sci/labs/orzuk/shaulytolk/GRASSY-Net

export PYTHONPATH=/sci/labs/orzuk/shaulytolk/GRASSY-Net:$PYTHONPATH
export XDG_CACHE_HOME=/tmp/cache_$SLURM_JOB_ID
export PIP_NO_CACHE_DIR=1
export WANDB_DIR=/sci/labs/orzuk/shaulytolk/GRASSY-Net/wandb
export WANDB_CACHE_DIR=/tmp/wandb_cache_$SLURM_JOB_ID

python -m moment_diffusion.train --config configs/QM9/qm9_drop09_cwd01_moment.yaml
