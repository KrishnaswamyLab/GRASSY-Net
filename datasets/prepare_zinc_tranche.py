"""
Prepare ZINC tranche dataset for GRASSY-DiT and GraphDIT training.
Reads SMILES from .smi file format and computes properties for all molecules.

Input format (.smi file):
    smiles zinc_id
    O=c1n(Cc2ccccc2)c(=O)n2c(=O)n(Cc3ccccc3)c(=O)n12 1652617
    ...

Outputs:
  - datasets/ZINC.npy (all molecules with properties)
  - datasets/ZINC_stats.npy (statistics for normalization)

Then run extract_scattering_fixed.py:
    python grassy_dit/extract_scattering_fixed.py --dataset datasets/ZINC.npy --stats datasets/ZINC_stats.npy --output grassy_dit/data_zinc --J 4 --moments 4

Usage:
    python -m datasets.prepare_zinc_tranche --input datasets/ZINC_tranches/BBAB/BBAB.smi --output_dir datasets/ZINC_tranches/BBAB

    # BBAB
    python -m datasets.prepare_zinc_tranche --input datasets/ZINC_tranches/BBAB/BBAB.smi --output_dir datasets/ZINC_tranches/BBAB --prefix BBAB

    # FBAB
    python -m datasets.prepare_zinc_tranche --input datasets/ZINC_tranches/FBAB/FBAB.smi --output_dir datasets/ZINC_tranches/FBAB --prefix FBAB

    # JBCD
    python -m datasets.prepare_zinc_tranche --input datasets/ZINC_tranches/JBCD/JBCD.smi --output_dir datasets/ZINC_tranches/JBCD --prefix JBCD
"""
import argparse
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm
from rdkit import Chem

from datasets.property_utils import PROPERTIES_TO_COMPUTE, compute_props


def read_smi_file(filepath):
    """
    Read a .smi file and return list of (smiles, zinc_id) tuples.
    
    Expected format (space or tab separated):
        smiles zinc_id
        O=c1n(Cc2ccccc2)... 1652617
        ...
    
    Handles files with or without header.
    """
    molecules = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Check if first line is a header
    first_line = lines[0].strip().split()
    if len(first_line) >= 1 and first_line[0].lower() in ['smiles', 'smile', 'smi']:
        start_idx = 1  # Skip header
    else:
        start_idx = 0
    
    for line in lines[start_idx:]:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) >= 2:
            smiles = parts[0]
            zinc_id = parts[1]
        elif len(parts) == 1:
            smiles = parts[0]
            zinc_id = f"mol_{len(molecules)}"
        else:
            continue
        
        molecules.append((smiles, zinc_id))
    
    return molecules


def main():
    parser = argparse.ArgumentParser(description='Prepare ZINC tranche dataset for GRASSY-DiT')
    parser.add_argument('--input', type=str, required=True,
                        help='Input .smi file path')
    parser.add_argument('--output_dir', type=str, default='datasets',
                        help='Output directory for .npy files (default: datasets)')
    parser.add_argument('--prefix', type=str, default='ZINC',
                        help='Prefix for output files (default: ZINC)')
    parser.add_argument('--max-molecules', type=int, default=None,
                        help='Maximum number of molecules to process (default: all)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ==========================================================================
    # Step 1: Load SMILES from .smi file
    # ==========================================================================
    print("="*60)
    print("Step 1: Loading SMILES from .smi file")
    print("="*60)
    
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    molecules = read_smi_file(input_path)
    print(f"Loaded {len(molecules)} molecules from {input_path}")
    
    if args.max_molecules and len(molecules) > args.max_molecules:
        molecules = molecules[:args.max_molecules]
        print(f"Limited to first {len(molecules)} molecules")

    # ==========================================================================
    # Step 2: Process all molecules and compute properties
    # ==========================================================================
    print("\n" + "="*60)
    print("Step 2: Computing properties for all molecules")
    print("="*60)
    
    out_dict = {}
    skipped = 0
    
    for smiles, zinc_id in tqdm(molecules, desc="Processing"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped += 1
            continue
        
        props = compute_props(mol)
        props['zinc_id'] = zinc_id
        out_dict[smiles] = props
    
    print(f"Processed {len(out_dict)} valid molecules (skipped {skipped} invalid)")
    
    # Save all molecules
    data_path = f'{args.output_dir}/{args.prefix}.npy'
    np.save(data_path, out_dict)
    print(f"Saved to {data_path}")

    # ==========================================================================
    # Step 3: Compute statistics
    # ==========================================================================
    print("\n" + "="*60)
    print("Step 3: Computing statistics")
    print("="*60)
    
    stats = {}
    
    for prop in PROPERTIES_TO_COMPUTE:
        values = [out_dict[smi][prop] for smi in out_dict.keys()
                  if prop in out_dict[smi] and not np.isnan(out_dict[smi][prop])]
        if values:
            stats[prop] = {'mean': float(np.mean(values)), 'std': float(np.std(values))}
            print(f"  {prop}: mean={stats[prop]['mean']:.3f}, std={stats[prop]['std']:.3f}")
        else:
            stats[prop] = {'mean': 0.0, 'std': 1.0}
            print(f"  {prop}: no valid values, using default mean=0.0, std=1.0")

    stats_path = f'{args.output_dir}/{args.prefix}_stats.npy'
    np.save(stats_path, stats)
    print(f"\nSaved statistics to {stats_path}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("DONE! Files created:")
    print("="*60)
    print(f"  - {data_path} ({len(out_dict)} molecules)")
    print(f"  - {stats_path}")
    print(f"\nNext: Extract scattering moments:")
    print(f"  python grassy_dit/extract_scattering_fixed.py --dataset {data_path} --stats {stats_path} --output grassy_dit/data/data_{args.prefix.lower()} --J 4 --moments 4")


if __name__ == "__main__":
    main()