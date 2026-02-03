#!/bin/bash
#SBATCH --job-name=guide_sweep
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=guide_sweep_%j.out
#SBATCH --error=guide_sweep_%j.err

echo "Starting guidance scale sweep..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
date

source molenv/bin/activate

python -u -m guided_diffusion.sweep_guidance_scale \
    --checkpoint runs/graphdit_qm9_43996065/graphdit_final.pt \
    --target "c1ccccc1" \
    --num_samples 100 \
    --num_nodes 9 \
    --scales 0.0 0.001 0.005 0.01 0.02 0.05

echo "Done!"
date
