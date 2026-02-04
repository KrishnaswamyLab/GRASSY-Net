"""
Prepare training data for the moment classifier.

Generates (noisy_X, noisy_E, timestep, node_mask, clean_moments) tuples
by applying noise at various timesteps to clean molecular graphs.

Usage:
    python -m guided_diffusion.prepare_classifier_data \
        --dit_checkpoint runs/graphdit_qm9_43996065/graphdit_final.pt \
        --data_dir data/qm9_classifier \
        --output_dir guided_diffusion/classifier_data \
        --num_timesteps_per_mol 10
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch_molecule import GraphDITMolecularGenerator
from torch_molecule.generator.graph_dit.utils import to_dense
from rdkit import Chem


def load_molecular_data(data_dir, csv_file='molecules.csv', scatter_file='scattering_moments.npy',
                        smiles_col='smiles', max_samples=None):
    """
    Load molecular data from directory.
    
    Args:
        data_dir: Directory containing molecules.csv and scattering_moments.npy
        csv_file: Name of CSV file with SMILES
        scatter_file: Name of scattering moments file
        smiles_col: Column name for SMILES in CSV
        max_samples: Maximum number of samples to load (None = all)
    
    Returns:
        smiles: List of SMILES strings
        scattering: numpy array of scattering moments [N, moment_dim]
    """
    df = pd.read_csv(os.path.join(data_dir, csv_file))
    smiles = df[smiles_col].tolist()
    scattering = np.load(os.path.join(data_dir, scatter_file))
    
    # Filter invalid molecules (dative bonds not supported)
    valid_smiles = []
    valid_scatter = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        has_dative = any(b.GetBondType() == Chem.BondType.DATIVE for b in mol.GetBonds())
        if not has_dative:
            valid_smiles.append(smi)
            valid_scatter.append(scattering[i])
    
    smiles = valid_smiles
    scattering = np.array(valid_scatter)
    
    # Limit samples if requested
    if max_samples is not None and len(smiles) > max_samples:
        indices = np.random.choice(len(smiles), max_samples, replace=False)
        smiles = [smiles[i] for i in indices]
        scattering = scattering[indices]
    
    return smiles, scattering


def prepare_classifier_dataset(
    dit_checkpoint: str,
    data_dir: str,
    output_dir: str,
    num_timesteps_per_mol: int = 10,
    max_samples: int = None,
    batch_size: int = 64,
    csv_file: str = 'molecules.csv',
    scatter_file: str = 'scattering_moments.npy',
    smiles_col: str = 'smiles',
):
    """
    Prepare classifier training data by generating noisy graphs at various timesteps.
    
    Uses a trained DiT checkpoint to get properly configured noise infrastructure.
    
    Args:
        dit_checkpoint: Path to trained DiT checkpoint
        data_dir: Directory with molecules.csv and scattering_moments.npy
        output_dir: Where to save the prepared dataset
        num_timesteps_per_mol: Number of timesteps to sample per molecule
        max_samples: Maximum molecules to process (None = all)
        batch_size: Batch size for processing
        csv_file: Name of CSV file
        scatter_file: Name of scattering file
        smiles_col: SMILES column name
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the trained DiT to get properly configured infrastructure
    print(f"Loading DiT from {dit_checkpoint}...")
    dit = GraphDITMolecularGenerator()
    dit.load_from_local(dit_checkpoint)
    
    # Get dataset info from the trained model
    dataset_info = dit.dataset_info
    active_index = dataset_info["active_index"]
    max_node = dit.max_node
    timesteps = dit.timesteps
    input_dim_X = dit.input_dim_X
    input_dim_E = dit.input_dim_E
    
    print(f"DiT config: max_node={max_node}, timesteps={timesteps}, Xdim={input_dim_X}, Edim={input_dim_E}")
    print(f"Active atom indices: {len(active_index)}")
    
    # Load molecular data
    print(f"\nLoading data from {data_dir}...")
    smiles, scattering = load_molecular_data(
        data_dir, csv_file, scatter_file, smiles_col, max_samples
    )
    print(f"Loaded {len(smiles)} molecules")
    print(f"Scattering dimension: {scattering.shape[-1]}")
    
    scattering_tensor = torch.tensor(scattering, dtype=torch.float32)
    
    # Convert SMILES to PyG dataset using DiT's method
    print("Converting SMILES to PyG format...")
    X_validated, _ = dit._validate_inputs(smiles, None)
    dataset = dit._convert_to_pytorch_data(X_validated, None)
    
    # Create dataloader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Storage for generated data
    all_noisy_X = []
    all_noisy_E = []
    all_timesteps = []
    all_node_masks = []
    all_clean_moments = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Move DiT components to device
    dit.transition_model = dit.transition_model.to(device)
    
    print(f"\nGenerating noisy samples ({num_timesteps_per_mol} timesteps per molecule)...")
    
    for batch_idx, batched_data in enumerate(tqdm(loader, desc="Processing batches")):
        batched_data = batched_data.to(device)
        
        # Convert to dense format
        data_x = F.one_hot(batched_data.x, num_classes=118).float()[:, active_index]
        data_edge_attr = F.one_hot(batched_data.edge_attr, num_classes=5).float()
        dense_data, node_mask = to_dense(
            data_x, batched_data.edge_index, data_edge_attr, 
            batched_data.batch, max_node
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E  # [B, N, Xdim], [B, N, N, Edim]
        
        B = X.shape[0]
        
        # Get clean scattering moments for this batch
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + B, len(scattering_tensor))
        clean_moments = scattering_tensor[start_idx:end_idx].to(device)
        
        # Sample multiple timesteps for each molecule in batch
        for _ in range(num_timesteps_per_mol):
            # Apply noise using DiT's method (samples random timesteps internally)
            noisy_data = dit.apply_noise(X, E, clean_moments, node_mask)
            
            # Store results
            all_noisy_X.append(noisy_data['X_t'].cpu())
            all_noisy_E.append(noisy_data['E_t'].cpu())
            all_timesteps.append(noisy_data['t'].cpu())
            all_node_masks.append(node_mask.cpu())
            all_clean_moments.append(clean_moments.cpu())
    
    # Concatenate all data
    print("\nConcatenating data...")
    noisy_X = torch.cat(all_noisy_X, dim=0)
    noisy_E = torch.cat(all_noisy_E, dim=0)
    timesteps_tensor = torch.cat(all_timesteps, dim=0)
    node_masks = torch.cat(all_node_masks, dim=0)
    clean_moments = torch.cat(all_clean_moments, dim=0)
    
    print(f"Total samples: {noisy_X.shape[0]}")
    print(f"noisy_X shape: {noisy_X.shape}")
    print(f"noisy_E shape: {noisy_E.shape}")
    print(f"timesteps shape: {timesteps_tensor.shape}")
    print(f"node_masks shape: {node_masks.shape}")
    print(f"clean_moments shape: {clean_moments.shape}")
    
    # Save dataset
    print(f"\nSaving to {output_dir}...")
    torch.save({
        'noisy_X': noisy_X,
        'noisy_E': noisy_E,
        'timesteps': timesteps_tensor,
        'node_masks': node_masks,
        'clean_moments': clean_moments,
        'metadata': {
            'num_samples': noisy_X.shape[0],
            'max_node': max_node,
            'Xdim': input_dim_X,
            'Edim': input_dim_E,
            'moment_dim': scattering.shape[-1],
            'num_timesteps': timesteps,
            'num_timesteps_per_mol': num_timesteps_per_mol,
            'source_data_dir': data_dir,
            'dit_checkpoint': dit_checkpoint,
        }
    }, os.path.join(output_dir, 'classifier_training_data.pt'))
    
    print("Done!")
    
    return {
        'num_samples': noisy_X.shape[0],
        'max_node': max_node,
        'Xdim': input_dim_X,
        'Edim': input_dim_E,
        'moment_dim': scattering.shape[-1],
    }


class ClassifierDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading prepared classifier training data.
    
    Usage:
        dataset = ClassifierDataset('guided_diffusion/classifier_data/classifier_training_data.pt')
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for noisy_X, noisy_E, t, node_mask, clean_moments in loader:
            pred = classifier(noisy_X, noisy_E, t, node_mask)
            loss = F.mse_loss(pred, clean_moments)
    """
    
    def __init__(self, data_path: str):
        """
        Args:
            data_path: Path to classifier_training_data.pt
        """
        data = torch.load(data_path)
        
        self.noisy_X = data['noisy_X']
        self.noisy_E = data['noisy_E']
        self.timesteps = data['timesteps']
        self.node_masks = data['node_masks']
        self.clean_moments = data['clean_moments']
        self.metadata = data['metadata']
        
    def __len__(self):
        return self.noisy_X.shape[0]
    
    def __getitem__(self, idx):
        return (
            self.noisy_X[idx],
            self.noisy_E[idx],
            self.timesteps[idx],
            self.node_masks[idx],
            self.clean_moments[idx],
        )
    
    @property
    def max_node(self):
        return self.metadata['max_node']
    
    @property
    def Xdim(self):
        return self.metadata['Xdim']
    
    @property
    def Edim(self):
        return self.metadata['Edim']
    
    @property
    def moment_dim(self):
        return self.metadata['moment_dim']
    
    @property
    def num_timesteps(self):
        return self.metadata['num_timesteps']


def main():
    parser = argparse.ArgumentParser(
        description='Prepare training data for moment classifier',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument('--dit_checkpoint', type=str, required=True,
                        help='Path to trained DiT checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing molecules.csv and scattering_moments.npy')
    parser.add_argument('--output_dir', type=str, default='guided_diffusion/classifier_data',
                        help='Output directory for prepared data')
    parser.add_argument('--num_timesteps_per_mol', type=int, default=10,
                        help='Number of timesteps to sample per molecule')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of molecules to process (None = all)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for processing')
    parser.add_argument('--csv_file', type=str, default='molecules.csv',
                        help='Name of CSV file with SMILES')
    parser.add_argument('--scatter_file', type=str, default='scattering_moments.npy',
                        help='Name of scattering moments file')
    parser.add_argument('--smiles_col', type=str, default='smiles',
                        help='Column name for SMILES in CSV')
    
    args = parser.parse_args()
    
    prepare_classifier_dataset(
        dit_checkpoint=args.dit_checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        num_timesteps_per_mol=args.num_timesteps_per_mol,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        csv_file=args.csv_file,
        scatter_file=args.scatter_file,
        smiles_col=args.smiles_col,
    )


if __name__ == '__main__':
    main()
