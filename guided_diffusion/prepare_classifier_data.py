"""
Prepare training data for the moment classifier.

Generates (noisy_X, noisy_E, timestep, node_mask, clean_moments) tuples
by applying noise at various timesteps to clean molecular graphs.

Usage:
    python -m guided_diffusion.prepare_classifier_data \
        --data_dir grassy_dit/data \
        --output_dir guided_diffusion/classifier_data \
        --num_timesteps_per_mol 10 \
        --max_samples 10000
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


class NoiseApplicator(GraphDITMolecularGenerator):
    """
    Minimal wrapper to access apply_noise() from GraphDITMolecularGenerator.
    
    We inherit from GraphDITMolecularGenerator to get access to:
    - apply_noise(): Adds discrete noise to graphs at specified timesteps
    - _convert_to_pytorch_data(): Converts SMILES to PyG format
    - transition_model, noise_schedule: Required for apply_noise
    """
    
    def __init__(self, max_node=50, Xdim=10, Edim=5):
        # Initialize with minimal config - we only need noise application
        super().__init__(
            hidden_size=256,  # Doesn't matter, we won't use the model
            num_layer=2,
            num_head=4,
            epochs=1,
            batch_size=32,
        )
        self.max_node = max_node
        self.input_dim_X = Xdim
        self.input_dim_E = Edim
        
    def setup_for_data(self, smiles_list, scattering):
        """
        Initialize internal state needed for apply_noise().
        
        This calls the parent's data processing to set up:
        - dataset_info (atom types, etc.)
        - transition_model
        - noise_schedule
        """
        # Don't pass scattering to _validate_inputs (it expects task labels)
        # Pass None for y, we'll handle scattering separately
        X, _ = self._validate_inputs(smiles_list, None)
        
        # Convert to PyG data (without properties)
        self._dataset = self._convert_to_pytorch_data(X, None)
        
        # Store scattering separately - we'll use it as labels
        self._scattering = torch.tensor(scattering, dtype=torch.float32)
        
        # Now we have everything needed for apply_noise()
        return self._dataset


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
    
    For each molecule:
    1. Load clean graph and scattering moments
    2. Sample K random timesteps
    3. Apply noise at each timestep using DiT's apply_noise()
    4. Save (noisy_X, noisy_E, t, node_mask, clean_moments)
    
    Args:
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
    
    print(f"Loading data from {data_dir}...")
    smiles, scattering = load_molecular_data(
        data_dir, csv_file, scatter_file, smiles_col, max_samples
    )
    print(f"Loaded {len(smiles)} molecules")
    print(f"Scattering dimension: {scattering.shape[-1]}")
    
    # Infer dimensions from scattering
    # scattering_dim = num_atom_types * num_levels * num_moments
    # Default: J=4 -> num_levels = 1 + 4 + 6 = 11, num_moments = 4
    scattering_dim = scattering.shape[-1]
    J = 4
    num_levels = 1 + J + J * (J - 1) // 2  # 11 for J=4
    num_moments = 4
    num_atom_types = scattering_dim // (num_levels * num_moments)
    
    print(f"Inferred num_atom_types: {num_atom_types}")
    
    # Initialize noise applicator
    # We need to determine max_node from the data
    max_atoms = 0
    for smi in smiles[:1000]:  # Sample to estimate
        mol = Chem.MolFromSmiles(smi)
        if mol:
            max_atoms = max(max_atoms, mol.GetNumAtoms())
    max_node = min(max_atoms + 10, 50)  # Add buffer, cap at 50
    print(f"Using max_node: {max_node}")
    
    noise_applicator = NoiseApplicator(max_node=max_node, Xdim=num_atom_types, Edim=5)
    
    # Set up data processing
    print("Setting up noise applicator...")
    dataset = noise_applicator.setup_for_data(smiles, scattering)
    
    # Get dataset info
    dataset_info = noise_applicator.dataset_info
    active_index = dataset_info["active_index"]
    timesteps = noise_applicator.timesteps
    
    print(f"Timesteps: {timesteps}")
    print(f"Active atom indices: {len(active_index)}")
    
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
    
    # Move noise applicator components to device
    noise_applicator.transition_model = noise_applicator.transition_model.to(device)
    
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
        
        # Clean scattering moments for this batch (from stored scattering, not batched_data.y)
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + B, len(noise_applicator._scattering))
        clean_moments = noise_applicator._scattering[start_idx:end_idx].to(device)
        
        B = X.shape[0]
        
        # Sample multiple timesteps for each molecule in batch
        for _ in range(num_timesteps_per_mol):
            # Sample random timesteps for each molecule
            t_int = torch.randint(0, timesteps, (B,), device=device)
            
            # Apply noise
            noisy_data = noise_applicator.apply_noise(X, E, clean_moments, node_mask)
            
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
            'Xdim': num_atom_types,
            'Edim': 5,
            'moment_dim': scattering_dim,
            'num_timesteps': timesteps,
            'num_timesteps_per_mol': num_timesteps_per_mol,
            'source_data_dir': data_dir,
        }
    }, os.path.join(output_dir, 'classifier_training_data.pt'))
    
    print("Done!")
    
    return {
        'num_samples': noisy_X.shape[0],
        'max_node': max_node,
        'Xdim': num_atom_types,
        'Edim': 5,
        'moment_dim': scattering_dim,
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
