#!/bin/bash

#SBATCH --job-name=BBAB_zinc_fixed
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --output=./logs/slurm/ZINC/%x_%j.out
#SBATCH --error=./logs/slurm/ZINC/%x_%j.err

cd ~/workspace/GRASSY-Net

ml uv
ml CUDA/12.1.1

source .venv/bin/activate

python train_grassy_fixed_scattering.py --config configs/ZINC/BBAB/BBAB_grassy.yaml