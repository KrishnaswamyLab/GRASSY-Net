"""
Stage 1: Data Preparation

Converts SMILES from .smi file to .npy format with computed molecular properties.
Wraps datasets/prepare_zinc_tranche.py functionality.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
from tqdm import tqdm
from rdkit import Chem

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets.property_utils import PROPERTIES_TO_COMPUTE, compute_props


def read_smi_file(filepath: str) -> list:
    """
    Read a .smi file and return list of (smiles, id) tuples.
    
    Handles files with or without header.
    """
    molecules = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Check if first line is a header
    first_line = lines[0].strip().split()
    if len(first_line) >= 1 and first_line[0].lower() in ['smiles', 'smile', 'smi']:
        start_idx = 1
    else:
        start_idx = 0
    
    for i, line in enumerate(lines[start_idx:]):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) >= 2:
            smiles = parts[0]
            mol_id = parts[1]
        elif len(parts) == 1:
            smiles = parts[0]
            mol_id = f"mol_{i}"
        else:
            continue
        
        molecules.append((smiles, mol_id))
    
    return molecules


def run_data_prep(
    input_file: str,
    output_dir: str,
    prefix: str = "dataset",
    max_molecules: int = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Run data preparation stage.
    
    Args:
        input_file: Path to input .smi file
        output_dir: Directory to save outputs
        prefix: Prefix for output files
        max_molecules: Maximum number of molecules to process (None = all)
    
    Returns:
        Tuple of (data_path, stats_path, metrics_dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Stage 1: Data Preparation")
    print("=" * 60)
    
    # Load SMILES
    print(f"\nLoading SMILES from: {input_file}")
    molecules = read_smi_file(input_file)
    print(f"Loaded {len(molecules)} molecules")
    
    if max_molecules and len(molecules) > max_molecules:
        molecules = molecules[:max_molecules]
        print(f"Limited to first {len(molecules)} molecules")
    
    # Process molecules and compute properties
    print("\nComputing molecular properties...")
    out_dict = {}
    skipped = 0
    
    for smiles, mol_id in tqdm(molecules, desc="Processing"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            skipped += 1
            continue
        
        props = compute_props(mol)
        props['mol_id'] = mol_id
        out_dict[smiles] = props
    
    print(f"Processed {len(out_dict)} valid molecules (skipped {skipped} invalid)")
    
    # Save molecules
    data_path = os.path.join(output_dir, f"{prefix}.npy")
    np.save(data_path, out_dict)
    print(f"Saved molecules to: {data_path}")
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = {}
    
    for prop in PROPERTIES_TO_COMPUTE:
        values = [
            out_dict[smi][prop] for smi in out_dict.keys()
            if prop in out_dict[smi] and not np.isnan(out_dict[smi][prop])
        ]
        if values:
            stats[prop] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values))
            }
            print(f"  {prop}: mean={stats[prop]['mean']:.3f}, std={stats[prop]['std']:.3f}")
        else:
            stats[prop] = {'mean': 0.0, 'std': 1.0}
            print(f"  {prop}: no valid values, using defaults")
    
    stats_path = os.path.join(output_dir, f"{prefix}_stats.npy")
    np.save(stats_path, stats)
    print(f"Saved statistics to: {stats_path}")
    
    # Metrics for reporting
    metrics = {
        'input_molecules': len(molecules),
        'valid_molecules': len(out_dict),
        'skipped_molecules': skipped,
        'properties_computed': list(PROPERTIES_TO_COMPUTE),
    }
    
    print(f"\nData preparation complete!")
    return data_path, stats_path, metrics
