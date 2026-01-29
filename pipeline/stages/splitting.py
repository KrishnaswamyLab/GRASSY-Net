"""
Stage 3: Data Splitting

Splits the dataset into train/val/test sets with deterministic seed.
Wraps grassy_dit/split_datasets.py functionality.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def filter_molecules(smiles_list: list, scattering_array: np.ndarray):
    """Filter out molecules with dative bonds (unsupported by torch.molecule)."""
    valid_smiles = []
    valid_scatter = []
    valid_indices = []
    
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        has_dative = any(b.GetBondType() == Chem.BondType.DATIVE for b in mol.GetBonds())
        if not has_dative:
            valid_smiles.append(smi)
            valid_scatter.append(scattering_array[i])
            valid_indices.append(i)
    
    return valid_smiles, np.array(valid_scatter), valid_indices


def split_data(smiles: list, scattering: np.ndarray, split_ratios: tuple, seed: int):
    """Split data into train/val/test sets using deterministic seed."""
    train_ratio, val_ratio, test_ratio = split_ratios
    indices = np.arange(len(smiles))
    
    # First split: separate test set
    if test_ratio > 0:
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_ratio, random_state=seed
        )
    else:
        train_val_idx, test_idx = indices, np.array([], dtype=int)
    
    # Second split: separate train and val
    if val_ratio > 0:
        adjusted_val_ratio = val_ratio / (train_ratio + val_ratio)
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=adjusted_val_ratio, random_state=seed
        )
    else:
        train_idx, val_idx = train_val_idx, np.array([], dtype=int)
    
    return train_idx, val_idx, test_idx


def save_split(
    name: str,
    indices: np.ndarray,
    smiles: list,
    scattering: np.ndarray,
    output_dir: str,
) -> None:
    """Save a single split to its own directory."""
    if len(indices) == 0:
        return
    
    split_dir = os.path.join(output_dir, name)
    os.makedirs(split_dir, exist_ok=True)
    
    split_smiles = [smiles[i] for i in indices]
    split_scatter = scattering[indices]
    
    # Save CSV
    csv_path = os.path.join(split_dir, "molecules.csv")
    pd.DataFrame({"smiles": split_smiles}).to_csv(csv_path, index=False)
    
    # Save scattering
    npy_path = os.path.join(split_dir, "scattering_moments.npy")
    np.save(npy_path, split_scatter)
    
    print(f"  {name}/: {len(indices)} molecules")


def run_splitting(
    scattering_path: str,
    molecules_csv_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    filter_dative: bool = True,
) -> Tuple[str, str, str, Dict[str, Any]]:
    """
    Run data splitting stage.
    
    Args:
        scattering_path: Path to scattering_moments.npy
        molecules_csv_path: Path to molecules.csv
        output_dir: Directory to save split data
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        seed: Random seed for REPRODUCIBLE splitting
        filter_dative: Whether to filter molecules with dative bonds
    
    Returns:
        Tuple of (train_dir, val_dir, test_dir, metrics_dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Stage 3: Data Splitting")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data...")
    df = pd.read_csv(molecules_csv_path)
    smiles_all = df['smiles'].tolist()
    scattering_all = np.load(scattering_path)
    
    print(f"  Loaded {len(smiles_all)} molecules")
    
    # Filter molecules
    if filter_dative:
        print("Filtering molecules (removing dative bonds)...")
        smiles, scattering, valid_indices = filter_molecules(smiles_all, scattering_all)
        print(f"  Kept {len(smiles)} / {len(smiles_all)} molecules")
    else:
        smiles = smiles_all
        scattering = scattering_all
        valid_indices = list(range(len(smiles_all)))
    
    # Validate ratios
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    
    # Split with deterministic seed
    print(f"\nSplitting with seed={seed} (ratios: {train_ratio}/{val_ratio}/{test_ratio})...")
    train_idx, val_idx, test_idx = split_data(
        smiles, scattering, (train_ratio, val_ratio, test_ratio), seed
    )
    
    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")
    print(f"  Test:  {len(test_idx)}")
    
    # Save splits
    print(f"\nSaving to: {output_dir}")
    save_split('train', train_idx, smiles, scattering, output_dir)
    save_split('val', val_idx, smiles, scattering, output_dir)
    save_split('test', test_idx, smiles, scattering, output_dir)
    
    # Save metadata
    metadata = {
        'seed': seed,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'total_molecules': len(smiles_all),
        'valid_molecules': len(smiles),
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'filtered_dative': filter_dative,
    }
    
    metadata_path = os.path.join(output_dir, 'split_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {metadata_path}")
    
    # Metrics
    metrics = metadata.copy()
    
    # Return paths
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    print(f"\nData splitting complete!")
    return train_dir, val_dir, test_dir, metrics
