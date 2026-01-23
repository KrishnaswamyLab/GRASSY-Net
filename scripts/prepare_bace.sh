#!/bin/bash

#SBATCH --job-name=prepare-bace
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=devel
#SBATCH --mem=256G
#SBATCH --output=./logs/slurm/%x_%j.out
#SBATCH --error=./logs/slurm/%x_%j.err

cd ~/workspace/GRASSY-Net

source .venv/bin/activate

python -m datasets.prepare_bace