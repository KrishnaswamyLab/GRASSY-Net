"""
Split a molecular dataset into train/val/test sets.
Saves split indices for reproducibility and creates separate directories.

Usage:
    python split_dataset.py --data_dir grassy_dit/data/moses --csv_file molecules.csv --scatter_file scattering.npy
    python split_dataset.py --data_dir grassy_dit/data/moses --split 0.8 0.1 0.1 --seed 42

    python -m grassy_dit.split_datasets --data_dir grassy_dit/data/data_bbab --split 0.8 0.1 0.1 --seed 42
"""

import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rdkit import Chem
import json


def filter_molecules(smiles_list, scattering_array):
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


def split_dataset(smiles, scattering, split_ratios, seed):
    """Split data into train/val/test sets."""
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


def save_split(name, indices, smiles, scattering, output_dir, csv_file, scatter_file, smiles_col):
    """Save a single split to its own directory."""
    if len(indices) == 0:
        return
    
    # Create subdirectory
    split_dir = os.path.join(output_dir, name)
    os.makedirs(split_dir, exist_ok=True)
    
    split_smiles = [smiles[i] for i in indices]
    split_scatter = scattering[indices]
    
    # Save CSV (same filename as original)
    split_df = pd.DataFrame({smiles_col: split_smiles})
    csv_path = os.path.join(split_dir, csv_file)
    split_df.to_csv(csv_path, index=False)
    
    # Save scattering (same filename as original)
    npy_path = os.path.join(split_dir, scatter_file)
    np.save(npy_path, split_scatter)
    
    print(f"  {name}/: {csv_file}, {scatter_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Split molecular dataset into train/val/test sets',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing the data files')
    parser.add_argument('--csv_file', type=str, default='molecules.csv',
                        help='CSV file with SMILES (default: molecules.csv)')
    parser.add_argument('--scatter_file', type=str, default='scattering_moments.npy',
                        help='Numpy file with scattering_moments features (default: scattering_moments.npy)')
    parser.add_argument('--smiles_col', type=str, default='smiles',
                        help='Column name for SMILES (default: smiles)')
    parser.add_argument('--split', type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        metavar=('TRAIN', 'VAL', 'TEST'),
                        help='Split ratios (default: 0.8 0.1 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: same as data_dir)')
    parser.add_argument('--no_filter', action='store_true',
                        help='Skip filtering molecules with dative bonds')
    
    args = parser.parse_args()
    
    # Validate split ratios
    if abs(sum(args.split) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(args.split)}")
    
    output_dir = args.output_dir or args.data_dir
    
    # =========================================================================
    # Load data
    # =========================================================================
    print(f"Loading data from: {args.data_dir}")
    df = pd.read_csv(os.path.join(args.data_dir, args.csv_file))
    smiles_all = df[args.smiles_col].tolist()
    scattering_all = np.load(os.path.join(args.data_dir, args.scatter_file))
    
    print(f"  Loaded {len(smiles_all)} molecules")
    
    # =========================================================================
    # Filter molecules
    # =========================================================================
    if args.no_filter:
        smiles = smiles_all
        scattering = scattering_all
        valid_indices = list(range(len(smiles_all)))
    else:
        print("Filtering molecules (removing dative bonds)...")
        smiles, scattering, valid_indices = filter_molecules(smiles_all, scattering_all)
        print(f"  Kept {len(smiles)} / {len(smiles_all)} molecules")
    
    # =========================================================================
    # Split
    # =========================================================================
    print(f"\nSplitting with ratios {args.split} (seed={args.seed})...")
    train_idx, val_idx, test_idx = split_dataset(
        smiles, scattering, args.split, args.seed
    )
    
    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")
    print(f"  Test:  {len(test_idx)}")
    
    # =========================================================================
    # Save split files into separate directories
    # =========================================================================
    print(f"\nSaving to: {output_dir}")
    
    save_split('train', train_idx, smiles, scattering, output_dir, 
               args.csv_file, args.scatter_file, args.smiles_col)
    save_split('val', val_idx, smiles, scattering, output_dir,
               args.csv_file, args.scatter_file, args.smiles_col)
    save_split('test', test_idx, smiles, scattering, output_dir,
               args.csv_file, args.scatter_file, args.smiles_col)
    
    # =========================================================================
    # Save split metadata
    # =========================================================================
    metadata = {
        'source_csv': args.csv_file,
        'source_scatter': args.scatter_file,
        'smiles_col': args.smiles_col,
        'split_ratios': args.split,
        'seed': args.seed,
        'filtered': not args.no_filter,
        'total_molecules': len(smiles_all),
        'valid_molecules': len(smiles),
        'train_size': len(train_idx),
        'val_size': len(val_idx),
        'test_size': len(test_idx),
        'train_indices': train_idx.tolist(),
        'val_indices': val_idx.tolist(),
        'test_indices': test_idx.tolist(),
    }
    
    metadata_path = os.path.join(output_dir, 'split_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata: {metadata_path}")
    
    print("\nDone!")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    print(f"  ├── train/")
    print(f"  │   ├── {args.csv_file}")
    print(f"  │   └── {args.scatter_file}")
    print(f"  ├── val/")
    print(f"  │   ├── {args.csv_file}")
    print(f"  │   └── {args.scatter_file}")
    print(f"  ├── test/")
    print(f"  │   ├── {args.csv_file}")
    print(f"  │   └── {args.scatter_file}")
    print(f"  └── split_metadata.json")


if __name__ == "__main__":
    main()