#!/bin/bash

#SBATCH --job-name=grassy
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=4
#SBATCH --partition=scavenge_gpu
#SBATCH --gpus=1
#SBATCH --mem=256G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err
cd ~/project/GRASSY-Net
module load miniconda
conda activate mfcn

python train_grassy.py