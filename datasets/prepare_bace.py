"""
Prepare BACE dataset for GRASSY-DiT and GraphDIT training.
Downloads BACE from PyTorch Geometric, class-balances, splits, and computes conditions.

Outputs:
  - datasets/BACE.npy (molecule data with properties)
  - datasets/BACE_stats.npy (statistics for normalization)
  - datasets/BACE_splits.json (train/val/test indices for reproducibility)
  - grassy_dit/data_bace/molecules.csv (SMILES list)
  - grassy_dit/data_bace/scattering_moments.npy (scattering features)

Usage:
    python datasets/prepare_bace.py
    python datasets/prepare_bace.py --J 11 --moments 4 --device cuda
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED

# For scattering extraction
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import MoleculeNet
from models.ScatteringTransform import GraphScatteringTransform
from datasets.load_ZINC_tranche import ZINCDataset


def compute_sas(mol):
    """Compute Synthetic Accessibility Score (1-10, lower is easier)."""
    try:
        from rdkit.Contrib.SA_Score import sascorer
        return float(sascorer.calculateScore(mol))
    except Exception:
        try:
            # Alternative location in some RDKit versions
            from rdkit.Chem import RDConfig
            import sys
            sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
            import sascorer
            return float(sascorer.calculateScore(mol))
        except Exception:
            return float('nan')


def compute_scs(mol):
    """Compute Synthetic Complexity Score using SCScore (1-5, lower is easier)."""
    try:
        # Try using scscore if available
        from rdkit.Chem import AllChem
        # SCScore is based on fingerprints - use a simple proxy based on complexity
        # Full SCScore requires a trained model, so we use BertzCT as approximation
        bertz = rdMolDescriptors.CalcBertzCT(mol)
        # Normalize to 1-5 range (typical BertzCT ranges from 0-2000+)
        scs = 1 + 4 * min(bertz / 2000, 1.0)
        return float(scs)
    except Exception:
        return float('nan')


def compute_props(mol, bace_label=None):
    """Compute properties for a molecule."""
    out = {}
    
    # SAS - Synthetic Accessibility Score
    out['SAS'] = compute_sas(mol)
    
    # SCS - Synthetic Complexity Score
    out['SCS'] = compute_scs(mol)
    
    # BACE activity label
    if bace_label is not None:
        out['bace_activity'] = float(bace_label)
    
    # Additional properties for evaluation
    try:
        out['qed'] = float(QED.qed(mol))
    except Exception:
        out['qed'] = float('nan')
    
    try:
        out['MolWt'] = float(Descriptors.MolWt(mol))
    except Exception:
        out['MolWt'] = float('nan')
    
    try:
        mw = out['MolWt']
        num_H = sum([atom.GetTotalNumHs() for atom in mol.GetAtoms()])
        out['HeavyAtomMolWt'] = mw - num_H * 1.00794 if not np.isnan(mw) else float('nan')
    except Exception:
        out['HeavyAtomMolWt'] = float('nan')
    
    try:
        out['TPSA'] = float(rdMolDescriptors.CalcTPSA(mol))
    except Exception:
        out['TPSA'] = float('nan')
    
    try:
        out['NumHAcceptors'] = float(rdMolDescriptors.CalcNumHBA(mol))
    except Exception:
        out['NumHAcceptors'] = float('nan')
    
    try:
        out['NumHDonors'] = float(rdMolDescriptors.CalcNumHBD(mol))
    except Exception:
        out['NumHDonors'] = float('nan')
    
    try:
        out['RingCount'] = float(mol.GetRingInfo().NumRings())
    except Exception:
        out['RingCount'] = float('nan')
    
    try:
        out['MolLogP'] = float(Descriptors.MolLogP(mol))
    except Exception:
        out['MolLogP'] = float('nan')
    
    return out


def extract_scattering(dataset_path, output_dir, stats_path, J, num_moments, batch_size, device):
    """Extract scattering coefficients for all molecules."""
    print(f"\n{'='*60}")
    print("Extracting scattering moments...")
    print(f"{'='*60}")
    
    # Setup device
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # Load dataset
    dataset = ZINCDataset(dataset_path, prop_stat_dict=stats_path, include_ki=False)
    print(f"Loaded {len(dataset)} molecules")
    print(f"Node features: {dataset.num_node_features}")

    # Create fixed scattering transform
    scattering = GraphScatteringTransform(
        in_channels=dataset.num_node_features,
        J=J,
        num_moments=num_moments,
    ).to(device)
    scattering.eval()

    print(f"\nScattering configuration:")
    print(f"  - Wavelet scales (J): {J}")
    print(f"  - Moments: {num_moments}")
    print(f"  - Output dimension: {scattering.out_shape()}")

    # Create dataloader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Extract scattering coefficients
    all_moments = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting scattering"):
            batch = batch.to(device)
            moments = scattering(batch)
            all_moments.append(moments.cpu().numpy())

    scattering_coeffs = np.vstack(all_moments)
    print(f"\nScattering coefficients shape: {scattering_coeffs.shape}")

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    scattering_path = os.path.join(output_dir, "scattering_moments.npy")
    molecules_path = os.path.join(output_dir, "molecules.csv")

    np.save(scattering_path, scattering_coeffs)
    pd.DataFrame({"smiles": dataset.smi}).to_csv(molecules_path, index=False)

    print(f"\nSaved to {output_dir}:")
    print(f"  - {scattering_path}")
    print(f"  - {molecules_path}")

    return scattering_coeffs


def main():
    parser = argparse.ArgumentParser(description='Prepare BACE dataset for GRASSY-DiT')
    parser.add_argument('--output-dir', type=str, default='datasets',
                        help='Output directory for .npy files (default: datasets)')
    parser.add_argument('--scatter-output', type=str, default='grassy_dit/data_bace',
                        help='Output directory for scattering (default: grassy_dit/data_bace)')
    parser.add_argument('--J', type=int, default=11, help='Wavelet scales (default: 11)')
    parser.add_argument('--moments', type=int, default=4, help='Statistical moments (default: 4)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'])
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
    # Step 4: Extract SMILES and compute properties
    # ==========================================================================
    print("\n" + "="*60)
    print("Step 4: Extracting SMILES and computing properties")
    print("="*60)
    
    out_dict = {}
    all_smiles = []
    skipped = 0
    
    for idx in tqdm(balanced_idx, desc="Processing molecules"):
        data = dataset[idx]
        smi = data.smiles if hasattr(data, 'smiles') else None
        
        # If SMILES not directly available, try to get from the dataset
        if smi is None:
            # MoleculeNet stores SMILES in the raw data
            try:
                raw_path = Path('./data/BACE/raw/bace.csv')
                if raw_path.exists():
                    df = pd.read_csv(raw_path)
                    smiles_col = 'mol' if 'mol' in df.columns else 'smiles'
                    smi = df.iloc[idx][smiles_col]
            except Exception:
                pass
        
        if smi is None:
            skipped += 1
            continue
            
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            skipped += 1
            continue
        
        bace_label = data.y.item()
        props = compute_props(mol, bace_label=bace_label)
        out_dict[smi] = props
        all_smiles.append(smi)

    # Save dataset
    dataset_path = f'{args.output_dir}/BACE.npy'
    np.save(dataset_path, out_dict)
    print(f"\nSaved {len(out_dict)} molecules to {dataset_path}")
    print(f"Skipped {skipped} invalid molecules")

    # ==========================================================================
    # Step 5: Compute statistics
    # ==========================================================================
    print("\n" + "="*60)
    print("Step 5: Computing statistics")
    print("="*60)
    
    prop_list = ['qed', 'HeavyAtomMolWt', 'MolWt', 'TPSA', 'NumHAcceptors', 
                 'NumHDonors', 'RingCount', 'MolLogP', 'SAS', 'SCS']
    stats = {}
    
    for prop in prop_list:
        values = [out_dict[smi][prop] for smi in out_dict.keys()
                  if prop in out_dict[smi] and not np.isnan(out_dict[smi][prop])]
        if values:
            stats[prop] = {'mean': np.mean(values), 'std': np.std(values)}
            print(f"  {prop}: mean={stats[prop]['mean']:.3f}, std={stats[prop]['std']:.3f}")
        else:
            stats[prop] = {'mean': 0.0, 'std': 1.0}

    stats_path = f'{args.output_dir}/BACE_stats.npy'
    np.save(stats_path, stats)
    print(f"\nSaved statistics to {stats_path}")

    # ==========================================================================
    # Step 6: Extract scattering moments
    # ==========================================================================
    extract_scattering(
        dataset_path=dataset_path,
        output_dir=args.scatter_output,
        stats_path=stats_path,
        J=args.J,
        num_moments=args.moments,
        batch_size=args.batch_size,
        device=args.device,
    )

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("DONE! Files created:")
    print("="*60)
    print(f"  - {dataset_path}")
    print(f"  - {stats_path}")
    print(f"  - {splits_path}")
    print(f"  - {args.scatter_output}/molecules.csv")
    print(f"  - {args.scatter_output}/scattering_moments.npy")
    print(f"\nDataset summary:")
    print(f"  - Total balanced molecules: {len(balanced_idx)}")
    print(f"  - Train: {len(train_idx)} (60%)")
    print(f"  - Val: {len(val_idx)} (20%)")
    print(f"  - Test: {len(test_idx)} (20%)")


if __name__ == "__main__":
    main()
