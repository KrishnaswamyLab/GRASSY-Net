#!/bin/bash
#SBATCH --job-name=clf_qm9
#SBATCH --partition=catfish
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=clf_qm9_%j.out
#SBATCH --error=clf_qm9_%j.err

source molenv/bin/activate
set -e

echo "=============================================="
echo "Classifier Guidance Training Pipeline (QM9)"
echo "Job ID: $SLURM_JOB_ID"
date
echo "=============================================="

DIT_CHECKPOINT="runs/graphdit_qm9_43996065/graphdit_final.pt"

echo "Step 0: Preparing QM9 scattering data..."
python -c "
import os, torch, numpy as np, pandas as pd
from torch_geometric.datasets import QM9
from tqdm import tqdm
from models.ScatteringTransform import GraphScatteringTransform
from evaluation.utils import compute_scattering_from_smiles

dataset = QM9(root='data/qm9')
smiles_list = [data.smiles for data in tqdm(dataset) if hasattr(data, 'smiles')][:10000]
print(f'Got {len(smiles_list)} SMILES')

scatter = GraphScatteringTransform(in_channels=4, J=4, num_moments=4).cuda()
moments = compute_scattering_from_smiles(smiles_list, scatter, ['C','N','O','F'], 'cuda', 64)
print(f'Moments shape: {moments.shape}')

valid_mask = torch.isfinite(moments).all(dim=1)
valid_smiles = [s for s, v in zip(smiles_list, valid_mask) if v]
valid_moments = moments[valid_mask].cpu().numpy()

os.makedirs('data/qm9_classifier', exist_ok=True)
pd.DataFrame({'smiles': valid_smiles}).to_csv('data/qm9_classifier/molecules.csv', index=False)
np.save('data/qm9_classifier/scattering_moments.npy', valid_moments)
print(f'Saved {len(valid_smiles)} molecules')
"

echo "Step 1: Generating noisy training samples..."
python -m guided_diffusion.prepare_classifier_data \
    --dit_checkpoint $DIT_CHECKPOINT \
    --data_dir data/qm9_classifier \
    --output_dir guided_diffusion/classifier_data \
    --num_timesteps_per_mol 10 --batch_size 64

echo "Step 2: Training moment classifier..."
python -m guided_diffusion.train_classifier \
    --data_path guided_diffusion/classifier_data/classifier_training_data.pt \
    --output_dir guided_diffusion/checkpoints/moment_classifier \
    --hidden_size 256 --num_layers 4 --num_heads 8 \
    --epochs 100 --batch_size 64 --wandb_project GRASSY-Classifier \
    --l1_lambda 1e-5

echo "Step 3: Testing classifier-guided generation..."
python -m guided_diffusion.generate_classifier_guided \
    --dit_checkpoint $DIT_CHECKPOINT \
    --classifier_checkpoint guided_diffusion/checkpoints/moment_classifier/checkpoint_best.pt \
    --target_smiles "c1ccccc1" --num_samples 100 --guidance_scale 1.0

echo "Done!"
date
