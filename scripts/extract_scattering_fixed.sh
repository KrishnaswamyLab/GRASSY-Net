#!/bin/bash

#SBATCH --job-name=extract_scattering_fixed
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --mem=32G
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

# Run generate_embeddings

python grassy_dit/extract_scattering_fixed.py --dataset datasets/MOSES.npy --stats datasets/MOSES_stats.npy  --output grassy_dit/data/moses