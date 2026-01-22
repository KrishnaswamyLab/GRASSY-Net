#!/bin/bash

#SBATCH --job-name=grassy_end_to_end
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=devel
#SBATCH --mem=128G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

# Exit on error
set -e

# Create log directory if it doesn't exist
mkdir -p ./logs/slurm

cd ~/workspace/GRASSY-Net

# Load modules
ml uv

# Activate virtual environment
source .venv/bin/activate

# Set number of workers to match CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run training
python train_grassy_end_to_end.py --config grassy_config.yaml