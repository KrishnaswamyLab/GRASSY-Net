"""Prepare MOSES dataset for GRASSY training."""
import argparse
import sys
from pathlib import Path

# Add external/moses to path before importing moses
sys.path.insert(0, str(Path(__file__).parent.parent / "external" / "moses"))

import numpy as np
import moses
from rdkit import Chem
from tqdm import tqdm
from datasets.property_utils import compute_props
from datasets.property_utils import PROPERTIES_TO_COMPUTE

##################################### added this to download the MOSES data if it is not present #####################################
import subprocess

# Auto-download MOSES data if missing or is LFS pointer
def ensure_moses_data():
    data_dir = Path(__file__).parent.parent / "external" / "moses" / "moses" / "dataset" / "data"
    train_file = data_dir / "train.csv.gz"
    
    # Check if file is a Git LFS pointer (small text file starting with "version")
    if train_file.exists():
        with open(train_file, 'rb') as f:
            header = f.read(7)
        if header == b'version':
            print(f"Detected Git LFS pointer, downloading actual data...")
            train_file.unlink()  # Remove pointer file
    
    if not train_file.exists():
        print(f"Downloading MOSES training data (~40MB)...")
        url = "https://media.githubusercontent.com/media/molecularsets/moses/master/moses/dataset/data/train.csv.gz"
        subprocess.run(["curl", "-L", "-o", str(train_file), url], check=True)
        print(f"Downloaded to {train_file}")

ensure_moses_data()
#####################################################################################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare MOSES dataset for GRASSY training.')
    parser.add_argument('--subset', type=int, default=None,
                        help='Number of molecules to use (default: all). E.g., --subset 12000')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for shuffling before subset (default: 42)')
    parser.add_argument('--output-dir', type=str, default='datasets',
                        help='Output directory (default: datasets)')
    args = parser.parse_args()

    print("Loading MOSES training set...")
    train_smiles = moses.get_dataset('train')
    print(f"Loaded {len(train_smiles)} SMILES")

    # Subset if requested
    if args.subset is not None:
        np.random.seed(args.seed)
        indices = np.random.permutation(len(train_smiles))[:args.subset]
        train_smiles = [train_smiles[i] for i in indices]
        print(f"Using random subset of {len(train_smiles)} molecules (seed={args.seed})")
        suffix = f"_{args.subset // 1000}K" if args.subset >= 1000 else f"_{args.subset}"
    else:
        suffix = ""

    print("Computing properties...")
    out_dict = {}
    skipped = 0

    for smi in tqdm(train_smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                skipped += 1
                continue
            out_dict[smi] = compute_props(mol, bace_label=None)
        except Exception as e:
            skipped += 1
            continue

    # Save
    output_path = f'{args.output_dir}/MOSES{suffix}.npy'
    np.save(output_path, out_dict)
    print(f"\nSaved {len(out_dict)} molecules to {output_path}")
    print(f"Skipped {skipped} invalid molecules")

    # Compute stats (for z-scoring)
    print("\nComputing statistics...")
    prop_list = PROPERTIES_TO_COMPUTE
    stats = {}
    for prop in prop_list:
        values = [out_dict[smi][prop] for smi in out_dict.keys()
                  if not np.isnan(out_dict[smi][prop])]
        if values:
            stats[prop] = {'mean': np.mean(values), 'std': np.std(values)}
        else:
            stats[prop] = {'mean': 0.0, 'std': 1.0}

    stats_path = f'{args.output_dir}/MOSES{suffix}_stats.npy'
    np.save(stats_path, stats)
    print(f"Saved statistics to {stats_path}")

