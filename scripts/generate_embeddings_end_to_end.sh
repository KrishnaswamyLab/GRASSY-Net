#!/bin/bash

#SBATCH --job-name=grassy_generate_embeddings
#SBATCH --time=4:00:00
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

# Run generate_embeddings
python -m notebooks.generate_embeddings_mlp --save_dir outputs/MOSES_12K_e2e_regress_nokld_2026-01-21-18-20-29 --model_path outputs/MOSES_12K_e2e_regress_nokld_2026-01-21-18-20-29/best-epoch=74-val_loss=0.005.ckpt --save_embeddings