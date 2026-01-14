#!/bin/bash

#SBATCH --job-name=grassy_scattering
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_h200
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd ~/workspace/GRASSY-Net

ml uv
ml CUDA/12.1.1

source .venv/bin/activate

python train_learnable_scattering.py