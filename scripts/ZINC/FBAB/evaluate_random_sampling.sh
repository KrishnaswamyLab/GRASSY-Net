#!/bin/bash

#SBATCH --job-name=evaluate_FBAB
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=./logs/slurm/ZINC/FBAB/%x_%j.out
#SBATCH --error=./logs/slurm/ZINC/FBAB/%x_%j.err

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

# Run eval
python -m evaluation.zinc_eval \
    --dataset grassy_dit/data/data_FBAB/test \
    --dit-checkpoint checkpoints/zinc/FBAB/checkpoint_best.pt \
    --grassy-checkpoint-dir ?? \
    --config ?? \
    --output-dir evaluation/results/ZINC/latent_sampling \
    --num-samples 1000