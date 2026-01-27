"""
Prepare BACE dataset for GRASSY-DiT and GraphDIT training.
Downloads BACE from PyTorch Geometric, class-balances, splits, and computes properties.

Outputs:
  - datasets/BACE_train.npy (train molecules with properties)
  - datasets/BACE_val.npy (val molecules with properties)
  - datasets/BACE_test.npy (test molecules with properties)
  - datasets/BACE_stats.npy (statistics for normalization)
  - datasets/BACE_splits.json (train/val/test indices for reproducibility)

Then run extract_scattering_fixed.py on each:
    python grassy_dit/extract_scattering_fixed.py --dataset datasets/BACE_train.npy --stats datasets/BACE_stats.npy --output grassy_dit/data_bace_train --J 11 --moments 4
    python grassy_dit/extract_scattering_fixed.py --dataset datasets/BACE_val.npy --stats datasets/BACE_stats.npy --output grassy_dit/data_bace_val --J 11 --moments 4
    python grassy_dit/extract_scattering_fixed.py --dataset datasets/BACE_test.npy --stats datasets/BACE_stats.npy --output grassy_dit/data_bace_test --J 11 --moments 4

Usage:
    python datasets/prepare_bace.py
    python datasets/prepare_bace.py --seed 42
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.datasets import MoleculeNet
from datasets.property_utils import PROPERTIES_TO_COMPUTE, compute_props
# At the top of your file, define a registry of available properties

def main():
    parser = argparse.ArgumentParser(description='Prepare BACE dataset for GRASSY-DiT')
    parser.add_argument('--output-dir', type=str, default='datasets',
                        help='Output directory for .npy files (default: datasets)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ==========================================================================
    # Step 1: Load BACE dataset from PyTorch Geometric
    # ==========================================================================
    print("="*60)
    print("Step 1: Loading BACE dataset from PyTorch Geometric")
    print("="*60)
    
    dataset = MoleculeNet(root='./data', name='BACE')
    print(f"Loaded {len(dataset)} molecules")
    
    # Get class distribution
    class_0_idx = [i for i, d in enumerate(dataset) if d.y.item() == 0]
    class_1_idx = [i for i, d in enumerate(dataset) if d.y.item() == 1]
    print(f"Class 0 (non-inhibitors): {len(class_0_idx)}")
    print(f"Class 1 (inhibitors): {len(class_1_idx)}")

    # ==========================================================================
    # Step 2: Class-balance by undersampling majority class
    # ==========================================================================
    print("\n" + "="*60)
    print("Step 2: Class-balancing by undersampling")
    print("="*60)
    
    minority_size = min(len(class_0_idx), len(class_1_idx))
    
    if len(class_0_idx) > len(class_1_idx):
        class_0_idx = np.random.choice(class_0_idx, minority_size, replace=False).tolist()
        print(f"Undersampled class 0 to {minority_size}")
    else:
        class_1_idx = np.random.choice(class_1_idx, minority_size, replace=False).tolist()
        print(f"Undersampled class 1 to {minority_size}")
    
    balanced_idx = class_0_idx + class_1_idx
    np.random.shuffle(balanced_idx)
    print(f"Balanced dataset size: {len(balanced_idx)}")

    # ==========================================================================
    # Step 3: Split into train/val/test (6:2:2)
    # ==========================================================================
    print("\n" + "="*60)
    print("Step 3: Splitting into train/val/test (6:2:2)")
    print("="*60)
    
    n = len(balanced_idx)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    
    train_idx = balanced_idx[:train_end]
    val_idx = balanced_idx[train_end:val_end]
    test_idx = balanced_idx[val_end:]
    
    print(f"Train: {len(train_idx)}")
    print(f"Val: {len(val_idx)}")
    print(f"Test: {len(test_idx)}")
    
    # Save splits to JSON for reproducibility
    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
        "seed": args.seed,
        "total_balanced": len(balanced_idx),
    }
    splits_path = f'{args.output_dir}/BACE_splits.json'
    with open(splits_path, 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits to {splits_path}")

    # ==========================================================================
    # Step 4: Load raw CSV to get SMILES
    # ==========================================================================
    print("\n" + "="*60)
    print("Step 4: Loading SMILES from raw CSV")
    print("="*60)
    
    raw_path = Path('./data/bace/raw/bace.csv')
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw BACE CSV not found at {raw_path}. Make sure PyTorch Geometric downloaded it.")
    
    df_raw = pd.read_csv(raw_path)
    smiles_col = 'mol' if 'mol' in df_raw.columns else 'smiles'
    print(f"Loaded {len(df_raw)} molecules from {raw_path}")

    # ==========================================================================
    # Step 5: Process each split and save separate .npy files
    # ==========================================================================
    print("\n" + "="*60)
    print("Step 5: Processing splits and computing properties")
    print("="*60)
    
    all_props = {}  # For computing overall stats
    
    for split_name, split_indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        print(f"\nProcessing {split_name} split ({len(split_indices)} molecules)...")
        
        out_dict = {}
        skipped = 0
        
        for idx in tqdm(split_indices, desc=f"  {split_name}"):
            data = dataset[idx]
            smi = df_raw.iloc[idx][smiles_col]
            
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                skipped += 1
                continue
            
            bace_label = data.y.item()
            props = compute_props(mol, bace_label=bace_label)
            out_dict[smi] = props
            all_props[smi] = props
        
        # Save this split
        split_path = f'{args.output_dir}/BACE_{split_name}.npy'
        np.save(split_path, out_dict)
        print(f"  Saved {len(out_dict)} molecules to {split_path} (skipped {skipped})")

    # ==========================================================================
    # Step 6: Compute statistics (from all data for consistent normalization)
    # ==========================================================================
    print("\n" + "="*60)
    print("Step 6: Computing statistics")
    print("="*60)
    
    prop_list = PROPERTIES_TO_COMPUTE

    stats = {}
    
    for prop in prop_list:
        values = [all_props[smi][prop] for smi in all_props.keys()
                  if prop in all_props[smi] and not np.isnan(all_props[smi][prop])]
        if values:
            stats[prop] = {'mean': np.mean(values), 'std': np.std(values)}
            print(f"  {prop}: mean={stats[prop]['mean']:.3f}, std={stats[prop]['std']:.3f}")
        else:
            stats[prop] = {'mean': 0.0, 'std': 1.0}

    stats_path = f'{args.output_dir}/BACE_stats.npy'
    np.save(stats_path, stats)
    print(f"\nSaved statistics to {stats_path}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("DONE! Files created:")
    print("="*60)
    print(f"  - {args.output_dir}/BACE_train.npy ({len(train_idx)} molecules)")
    print(f"  - {args.output_dir}/BACE_val.npy ({len(val_idx)} molecules)")
    print(f"  - {args.output_dir}/BACE_test.npy ({len(test_idx)} molecules)")
    print(f"  - {stats_path}")
    print(f"  - {splits_path}")
    print(f"\nNext: Extract scattering moments for each split:")
    print(f"  python grassy_dit/extract_scattering_fixed.py --dataset datasets/BACE_train.npy --stats datasets/BACE_stats.npy --output grassy_dit/data_bace_train --J 4 --moments 4")
    print(f"  python grassy_dit/extract_scattering_fixed.py --dataset datasets/BACE_val.npy --stats datasets/BACE_stats.npy --output grassy_dit/data_bace_val --J 4 --moments 4")
    print(f"  python grassy_dit/extract_scattering_fixed.py --dataset datasets/BACE_test.npy --stats datasets/BACE_stats.npy --output grassy_dit/data_bace_test --J 4 --moments 4")


if __name__ == "__main__":
    main()
