"""
Baseline GraphDiT evaluation script (no cross-attention, no scattering conditioning).

Evaluates the baseline GraphDiT model (GraphDITMolecularGenerator) on QM9 or other datasets.
Generates molecules unconditionally and computes MOSES-style metrics.

Metrics: Validity, Uniqueness, Novelty, Diversity, FCD.

Usage:
    python -m evaluation.baseline_eval \
        --checkpoint runs/graphdit_qm9_43996065/graphdit_final.pt \
        --test-dir runs/qm9_h512_l1/stage_3_splitting/test \
        --train-dir runs/qm9_h512_l1/stage_3_splitting/train \
        --num-samples 1000
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from torch_molecule import GraphDITMolecularGenerator

# Output directory
RESULTS_DIR = Path(__file__).parent / "results" / "baseline"

# Reference heavy atom types
REFERENCE_ATOM_TYPES = ["C", "N", "O", "F"]


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_test_data(test_dir: str, smiles_col: str = "smiles") -> List[str]:
    """Load test SMILES from a directory."""
    test_path = Path(test_dir)
    csv_path = test_path / "molecules.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"molecules.csv not found in {test_dir}")
    smiles = pd.read_csv(csv_path)[smiles_col].tolist()
    return smiles


def load_train_data(train_dir: str, smiles_col: str = "smiles") -> List[str]:
    """Load train SMILES for novelty computation."""
    train_path = Path(train_dir)
    csv_path = train_path / "molecules.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"molecules.csv not found in {train_dir}")
    smiles = pd.read_csv(csv_path)[smiles_col].tolist()
    return smiles


# -----------------------------------------------------------------------------
# Generation Functions
# -----------------------------------------------------------------------------

def generate_unconditional(
    model: GraphDITMolecularGenerator,
    num_samples: int,
    batch_size: int,
    device: str,
) -> tuple[List[str], Dict]:
    """Generate molecules unconditionally."""
    print(f"\nGenerating {num_samples} molecules (unconditional)...")

    all_smiles = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_samples)
        current_batch_size = batch_end - batch_start

        # Generate without any conditioning
        smiles_batch = model.generate(
            num_nodes=None,  # Sample from training distribution
            batch_size=current_batch_size,
        )

        valid_smiles = [s for s in smiles_batch if s is not None]
        all_smiles.extend(valid_smiles)

    stats = {
        "mode": "unconditional",
        "num_generated": len(all_smiles),
        "num_attempted": num_samples,
        "success_rate": len(all_smiles) / num_samples if num_samples > 0 else 0,
    }

    print(f"Generated {len(all_smiles)} valid molecules ({stats['success_rate']:.1%})")
    return all_smiles, stats


# -----------------------------------------------------------------------------
# Metrics (reused from zinc_eval_direct.py)
# -----------------------------------------------------------------------------

def get_valid_mols(smiles_list: List[str]) -> List:
    """Convert SMILES to valid RDKit mol objects."""
    return [Chem.MolFromSmiles(smi) for smi in smiles_list if Chem.MolFromSmiles(smi)]


def get_morgan_fingerprint(mol, radius: int = 2, n_bits: int = 2048):
    """Get Morgan fingerprint for a molecule."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def compute_validity(smiles_list: List[str]) -> Dict:
    """Compute validity metrics."""
    if not smiles_list:
        return {"validity": 0.0, "n_valid": 0}

    valid_mols = get_valid_mols(smiles_list)
    validity = len(valid_mols) / len(smiles_list)

    return {
        "validity": validity,
        "n_valid": len(valid_mols),
    }


def compute_uniqueness(smiles_list: List[str]) -> Dict:
    """Compute uniqueness metrics."""
    valid_smiles = [Chem.MolToSmiles(mol) for mol in get_valid_mols(smiles_list)]

    if not valid_smiles:
        return {"uniqueness": 0.0, "n_unique": 0}

    unique_smiles = set(valid_smiles)
    return {
        "uniqueness": len(unique_smiles) / len(valid_smiles),
        "n_unique": len(unique_smiles),
    }


def compute_novelty(smiles_list: List[str], reference_smiles: List[str]) -> Dict:
    """Compute novelty (fraction not in reference set)."""
    def canonicalize(smi: str):
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol) if mol else None

    reference_set = {canonicalize(smi) for smi in reference_smiles if canonicalize(smi)}

    novel_count = 0
    valid_count = 0
    for smi in smiles_list:
        canon = canonicalize(smi)
        if canon:
            valid_count += 1
            if canon not in reference_set:
                novel_count += 1

    return {
        "novelty": novel_count / valid_count if valid_count > 0 else 0.0,
        "n_novel": novel_count,
    }


def compute_diversity(smiles_list: List[str], sample_size: int = 5000) -> float:
    """Compute internal diversity using Tanimoto distance."""
    valid_mols = get_valid_mols(smiles_list)

    if len(valid_mols) < 2:
        return 0.0

    if len(valid_mols) > sample_size:
        indices = np.random.choice(len(valid_mols), sample_size, replace=False)
        valid_mols = [valid_mols[i] for i in indices]

    fps = [get_morgan_fingerprint(mol) for mol in valid_mols]

    total_distance = 0.0
    count = 0
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            total_distance += 1 - DataStructs.TanimotoSimilarity(fps[i], fps[j])
            count += 1

    return total_distance / count if count > 0 else 0.0


def compute_fcd(smiles_list: List[str], reference_smiles: List[str], device: str = "cpu") -> float:
    """Compute Frechet ChemNet Distance."""
    from fcd_torch import FCD as FCDMetric

    gen_smiles = [Chem.MolToSmiles(mol) for mol in get_valid_mols(smiles_list)]
    ref_smiles = [Chem.MolToSmiles(mol) for mol in get_valid_mols(reference_smiles)]

    if not gen_smiles:
        return float("inf")

    fcd_metric = FCDMetric(device=device)
    return float(fcd_metric(gen=gen_smiles, ref=ref_smiles))


def compute_all_metrics(generated_smiles: List[str], reference_smiles: List[str], device: str = "cpu") -> Dict:
    """Compute all evaluation metrics."""
    print("\nComputing metrics...")
    metrics = {}

    print("  Validity...")
    metrics.update(compute_validity(generated_smiles))

    print("  Uniqueness...")
    metrics.update(compute_uniqueness(generated_smiles))

    print("  Novelty...")
    metrics.update(compute_novelty(generated_smiles, reference_smiles))

    print("  Diversity...")
    metrics["diversity"] = compute_diversity(generated_smiles)

    print("  FCD...")
    metrics["fcd"] = compute_fcd(generated_smiles, reference_smiles, device)

    return metrics


# -----------------------------------------------------------------------------
# Output Formatting
# -----------------------------------------------------------------------------

def format_results_table(metrics: Dict) -> str:
    """Format metrics as a readable table."""
    rows = ["| Metric | Value |", "|--------|-------|"]

    display_metrics = [
        ("Validity", "validity"),
        ("Uniqueness", "uniqueness"),
        ("Novelty", "novelty"),
        ("Diversity", "diversity"),
        ("FCD", "fcd"),
    ]

    for display_name, key in display_metrics:
        value = metrics.get(key, "-")
        if isinstance(value, float):
            rows.append(f"| {display_name} | {value:.4f} |")
        else:
            rows.append(f"| {display_name} | {value} |")

    return "\n".join(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Baseline GraphDiT evaluation (no cross-attention, no scattering)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required arguments
    parser.add_argument("--checkpoint", required=True, help="Path to baseline GraphDiT checkpoint")
    parser.add_argument("--test-dir", required=True, help="Directory with test molecules.csv")
    parser.add_argument("--train-dir", required=True, help="Directory with train molecules.csv (for novelty)")

    # Generation arguments
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of molecules to generate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")

    # Output arguments
    parser.add_argument("--output-dir", default=None, help="Output directory (default: evaluation/results/baseline/)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smiles-col", default="smiles", help="SMILES column name in CSV")

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load test and train data
    print(f"Loading test data from {args.test_dir}...")
    test_smiles = load_test_data(args.test_dir, args.smiles_col)
    print(f"Loaded {len(test_smiles)} test molecules")

    print(f"Loading train data from {args.train_dir}...")
    train_smiles = load_train_data(args.train_dir, args.smiles_col)
    print(f"Loaded {len(train_smiles)} train molecules")

    # Load baseline model
    print(f"\nLoading baseline GraphDiT model from {args.checkpoint}...")
    model = GraphDITMolecularGenerator(device=args.device)
    model.load_from_local(args.checkpoint)

    # Generate molecules
    generated_smiles, gen_stats = generate_unconditional(
        model, args.num_samples, args.batch_size, args.device
    )

    # Compute metrics
    metrics = compute_all_metrics(generated_smiles, test_smiles, args.device)

    # Print results
    print(f"\n{'='*60}")
    print(f"Results (Baseline GraphDiT)")
    print(f"{'='*60}")
    print(format_results_table(metrics))

    # Save generated samples
    samples_path = output_dir / f"generated_baseline_{timestamp}.txt"
    with open(samples_path, "w") as f:
        for smi in generated_smiles:
            f.write(smi + "\n")
    print(f"\nSaved samples to {samples_path}")

    # Save results
    results = {
        "config": vars(args),
        "generation_stats": gen_stats,
        "metrics": metrics,
        "timestamp": timestamp,
    }

    # Convert numpy types for JSON
    def convert_to_native(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        return obj

    results_path = output_dir / f"baseline_eval_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(convert_to_native(results), f, indent=2)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
