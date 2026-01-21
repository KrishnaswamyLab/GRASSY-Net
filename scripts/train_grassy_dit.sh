#!/bin/bash

#SBATCH --job-name=grassy_dit_fixed_scattering
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
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
ml CUDA/12.1.1

# Activate virtual environment
source .venv/bin/activate

nvidia-smi

# Verify GPU is available (helpful for debugging)
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# Set number of workers to match CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run training
python -m grassy_dit.train --data_dir grassy_dit/data/moses_12k --epochs 100 --checkpoint_dir ./checkpoints 