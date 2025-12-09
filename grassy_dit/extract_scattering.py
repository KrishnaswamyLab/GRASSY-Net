"""
Extract scattering moments from trained GRASSY model.
Outputs: scattering_moments.npy and molecules.csv for train.py
"""
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from models.LEGS_module import Scatter
from datasets.load_ZINC_tranche import ZINCDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset and trained scatter model
dataset = ZINCDataset('datasets/fields_1.npy', prop_stat_dict='datasets/fields_1_stats.npy', include_ki=False)
scatter = Scatter(in_channels=dataset.num_node_features, trainable_laziness=False).to(device)
scatter.load_state_dict(torch.load('scripts/trained_models/fields_1_trained.npy', map_location=device))
scatter.eval()

# Extract moments and SMILES
loader = DataLoader(dataset, batch_size=1, shuffle=False)
all_moments = []
all_smiles = []

with torch.no_grad():
    for i, data in enumerate(loader):
        data = data.to(device)
        moments, _ = scatter(data)
        all_moments.append(moments.cpu().numpy())
        # Get SMILES from dataset directly
        all_smiles.append(dataset[i].smiles if hasattr(dataset[i], 'smiles') else str(i))

# Save
scattering = np.vstack(all_moments)
print(f"Scattering moments: {scattering.shape}")
print(f"SMILES count: {len(all_smiles)}")
print(f"First SMILES: {all_smiles[0]}")

np.save('grassy_dit/data/scattering_moments.npy', scattering)
pd.DataFrame({'smiles': all_smiles}).to_csv('grassy_dit/data/molecules.csv', index=False)
print("Saved to grassy_dit/data/")
