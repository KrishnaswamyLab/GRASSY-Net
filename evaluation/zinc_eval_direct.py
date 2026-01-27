"""
Direct evaluation script for GRASSY-DiT on ZINC tranches.

Unlike zinc_eval.py (which requires GRASSY VAE), this script:
1. Conditions DIRECTLY on test set scattering moments (no VAE needed)
2. Supports reconstruction evaluation (with/without moments)
3. Supports scaffold-constrained reconstruction

Metrics: Validity, Uniqueness, Novelty, Coverage, Diversity, Similarity, FCD.

Usage:
    # Basic evaluation - generate conditioned on test scattering
    python -m evaluation.zinc_eval_direct \
        --checkpoint checkpoints/zinc/fbab/checkpoint_best.pt \
        --test-dir grassy_dit/data/data_fbab/test \
        --config configs/ZINC/FBAB/FBAB_dit_config.yaml \
        --num-samples 1000

    # With reconstruction evaluation
    python -m evaluation.zinc_eval_direct \
        --checkpoint checkpoints/zinc/fbab/checkpoint_best.pt \
        --test-dir grassy_dit/data/data_fbab/test \
        --config configs/ZINC/FBAB/FBAB_dit_config.yaml \
        --num-samples 1000 \
        --recon-samples 50 \
        --recon-attempts 3

    # Unconditional generation (baseline)
    python -m evaluation.zinc_eval_direct \
        --checkpoint checkpoints/zinc/fbab/checkpoint_best.pt \
        --test-dir grassy_dit/data/data_fbab/test \
        --config configs/ZINC/FBAB/FBAB_dit_config.yaml \
        --num-samples 1000 \
        --unconditional
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "moses"))

from grassy_dit.sample import load_model_from_checkpoint, build_dummy_scattering, smiles_to_scaffold

# Output directory
RESULTS_DIR = Path(__file__).parent / "results" / "ZINC"

# Reference heavy atom types
REFERENCE_ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I"]


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_test_data(test_dir: str, smiles_col: str = "smiles") -> Tuple[List[str], np.ndarray]:
    """Load test SMILES and scattering moments from a directory."""
    test_path = Path(test_dir)
    
    # Load SMILES
    csv_path = test_path / "molecules.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"molecules.csv not found in {test_dir}")
    smiles = pd.read_csv(csv_path)[smiles_col].tolist()
    
    # Load scattering
    scatter_path = test_path / "scattering_moments.npy"
    if not scatter_path.exists():
        raise FileNotFoundError(f"scattering_moments.npy not found in {test_dir}")
    scattering = np.load(scatter_path)
    
    assert len(smiles) == len(scattering), f"Mismatch: {len(smiles)} SMILES vs {len(scattering)} scattering"
    
    return smiles, scattering


# -----------------------------------------------------------------------------
# Generation Functions
# -----------------------------------------------------------------------------

def generate_conditioned(
    model,
    test_scattering: np.ndarray,
    num_samples: int,
    batch_size: int,
    device: str,
) -> Tuple[List[str], Dict]:
    """Generate molecules conditioned on test scattering moments."""
    print(f"\nGenerating {num_samples} molecules (conditioned on test scattering)...")
    
    all_smiles = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_samples)
        current_batch_size = batch_end - batch_start
        
        # Sample random scattering from test set
        indices = np.random.choice(len(test_scattering), current_batch_size, replace=True)
        batch_scattering = torch.tensor(
            test_scattering[indices],
            dtype=torch.float32,
            device=device,
        )
        
        # Generate
        smiles_batch = model.generate(
            scattering=batch_scattering,
            num_nodes=None,
            batch_size=current_batch_size,
        )
        
        valid_smiles = [s for s in smiles_batch if s is not None]
        all_smiles.extend(valid_smiles)
    
    stats = {
        "mode": "conditioned",
        "num_generated": len(all_smiles),
        "num_attempted": num_samples,
        "success_rate": len(all_smiles) / num_samples if num_samples > 0 else 0,
    }
    
    print(f"Generated {len(all_smiles)} valid molecules ({stats['success_rate']:.1%})")
    return all_smiles, stats


def generate_unconditional(
    model,
    num_samples: int,
    batch_size: int,
    device: str,
) -> Tuple[List[str], Dict]:
    """Generate molecules unconditionally (CFG with guide_scale=0)."""
    print(f"\nGenerating {num_samples} molecules (unconditional)...")
    
    # Build dummy scattering
    dummy = build_dummy_scattering(model)
    
    all_smiles = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # Save and set guide_scale
    original_guide_scale = getattr(model, "guide_scale", 2.0)
    model.guide_scale = 0.0
    
    try:
        for batch_idx in tqdm(range(num_batches), desc="Generating"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, num_samples)
            current_batch_size = batch_end - batch_start
            
            batch_scattering = torch.tensor(
                np.repeat(dummy[None, :], current_batch_size, axis=0),
                dtype=torch.float32,
                device=device,
            )
            
            smiles_batch = model.generate(
                scattering=batch_scattering,
                num_nodes=None,
                batch_size=current_batch_size,
            )
            
            valid_smiles = [s for s in smiles_batch if s is not None]
            all_smiles.extend(valid_smiles)
    finally:
        model.guide_scale = original_guide_scale
    
    stats = {
        "mode": "unconditional",
        "num_generated": len(all_smiles),
        "num_attempted": num_samples,
        "success_rate": len(all_smiles) / num_samples if num_samples > 0 else 0,
    }
    
    print(f"Generated {len(all_smiles)} valid molecules ({stats['success_rate']:.1%})")
    return all_smiles, stats


# -----------------------------------------------------------------------------
# Reconstruction Evaluation
# -----------------------------------------------------------------------------

def canonicalize(smiles: str) -> Optional[str]:
    """Canonicalize SMILES, return None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def reconstruction_eval(
    model,
    test_smiles: List[str],
    test_scattering: np.ndarray,
    num_molecules: int,
    attempts: int,
    device: str,
) -> Dict:
    """
    Test reconstruction: compare with_moments vs without_moments (unconditional).
    
    For each test molecule:
    1. Generate `attempts` molecules conditioned on its scattering
    2. Generate `attempts` molecules unconditionally (same num_atoms)
    3. Check if any generation exactly matches the reference
    """
    print(f"\nRunning reconstruction eval on {num_molecules} molecules ({attempts} attempts each)...")
    
    indices = np.random.choice(len(test_smiles), min(num_molecules, len(test_smiles)), replace=False)
    
    results = {
        "with_moments": 0,
        "without_moments": 0,
        "total": 0,
    }
    
    original_guide_scale = getattr(model, "guide_scale", 2.0)
    
    for idx in tqdm(indices, desc="Reconstruction eval"):
        ref_smiles = test_smiles[idx]
        ref_mol = Chem.MolFromSmiles(ref_smiles)
        if ref_mol is None:
            continue
        
        ref_canon = Chem.MolToSmiles(ref_mol)
        num_atoms = ref_mol.GetNumAtoms()
        results["total"] += 1
        
        # Prepare scattering batch
        scattering = torch.tensor(
            np.repeat(test_scattering[idx:idx+1], attempts, axis=0),
            dtype=torch.float32,
            device=device,
        )
        
        # With moments (conditioned)
        model.guide_scale = original_guide_scale
        gen_with = model.generate(
            scattering=scattering,
            num_nodes=num_atoms,
            batch_size=attempts,
        )
        if any(canonicalize(s) == ref_canon for s in gen_with if s):
            results["with_moments"] += 1
        
        # Without moments (unconditional)
        model.guide_scale = 0.0
        gen_without = model.generate(
            scattering=scattering,
            num_nodes=num_atoms,
            batch_size=attempts,
        )
        if any(canonicalize(s) == ref_canon for s in gen_without if s):
            results["without_moments"] += 1
    
    model.guide_scale = original_guide_scale
    
    total = results["total"]
    return {
        "num_molecules": total,
        "attempts_per_molecule": attempts,
        "with_moments_count": results["with_moments"],
        "without_moments_count": results["without_moments"],
        "with_moments_rate": results["with_moments"] / total if total > 0 else 0,
        "without_moments_rate": results["without_moments"] / total if total > 0 else 0,
    }


def scaffold_reconstruction_eval_single(
    model,
    test_smiles: List[str],
    test_scattering: np.ndarray,
    mol_indices: np.ndarray,
    attempts: int,
    device: str,
    num_atoms_to_remove: int,
) -> Dict:
    """
    Test scaffold-constrained reconstruction for a specific number of atoms to remove.
    
    For each test molecule:
    1. Remove `num_atoms_to_remove` atoms (evenly spaced indices)
    2. Generate with scaffold constraint + scattering moments
    3. Generate with scaffold constraint but NO moments (unconditional)
    4. Check if any generation exactly matches the reference
    """
    results = {
        "with_moments": 0,
        "without_moments": 0,
        "total": 0,
        "skipped": 0,
    }
    
    original_guide_scale = getattr(model, "guide_scale", 2.0)
    
    for idx in mol_indices:
        ref_smiles = test_smiles[idx]
        ref_mol = Chem.MolFromSmiles(ref_smiles)
        if ref_mol is None:
            results["skipped"] += 1
            continue
        
        num_atoms = ref_mol.GetNumAtoms()
        
        # Generate evenly spaced indices to remove
        if num_atoms_to_remove >= num_atoms:
            results["skipped"] += 1
            continue
        
        # Evenly space the atoms to remove across the molecule
        remove_indices = [int(i * num_atoms / (num_atoms_to_remove + 1)) for i in range(1, num_atoms_to_remove + 1)]
        
        ref_canon = Chem.MolToSmiles(ref_mol)
        results["total"] += 1
        
        # Prepare scaffold tensors
        try:
            scaffold_X, scaffold_E, scaffold_mask, node_mask, n_atoms = smiles_to_scaffold(
                ref_smiles,
                model.max_node,
                model.dataset_info["atom_decoder"],
                model.dataset_info.get("bond_decoder", None),
                scaffold_pattern=None,
                remove_indices=remove_indices,
                num_nodes=num_atoms,
            )
        except Exception as e:
            results["skipped"] += 1
            results["total"] -= 1
            continue
        
        # Expand for batch
        scaffold_X_batch = scaffold_X.unsqueeze(0).expand(attempts, -1, -1)
        scaffold_E_batch = scaffold_E.unsqueeze(0).expand(attempts, -1, -1, -1)
        scaffold_mask_batch = scaffold_mask.unsqueeze(0).expand(attempts, -1)
        
        # Prepare scattering batch
        scattering = torch.tensor(
            np.repeat(test_scattering[idx:idx+1], attempts, axis=0),
            dtype=torch.float32,
            device=device,
        )
        
        # With moments (conditioned) + scaffold
        model.guide_scale = original_guide_scale
        gen_with = model.generate(
            scattering=scattering,
            num_nodes=num_atoms,
            batch_size=attempts,
            scaffold_X=scaffold_X_batch,
            scaffold_E=scaffold_E_batch,
            scaffold_node_mask=scaffold_mask_batch,
        )
        if any(canonicalize(s) == ref_canon for s in gen_with if s):
            results["with_moments"] += 1
        
        # Without moments (unconditional) + scaffold
        model.guide_scale = 0.0
        gen_without = model.generate(
            scattering=scattering,
            num_nodes=num_atoms,
            batch_size=attempts,
            scaffold_X=scaffold_X_batch,
            scaffold_E=scaffold_E_batch,
            scaffold_node_mask=scaffold_mask_batch,
        )
        if any(canonicalize(s) == ref_canon for s in gen_without if s):
            results["without_moments"] += 1
    
    model.guide_scale = original_guide_scale
    
    total = results["total"]
    return {
        "num_atoms_removed": num_atoms_to_remove,
        "num_molecules": total,
        "skipped": results["skipped"],
        "with_moments_count": results["with_moments"],
        "without_moments_count": results["without_moments"],
        "with_moments_rate": results["with_moments"] / total if total > 0 else 0,
        "without_moments_rate": results["without_moments"] / total if total > 0 else 0,
    }


def scaffold_reconstruction_eval(
    model,
    test_smiles: List[str],
    test_scattering: np.ndarray,
    num_molecules: int,
    attempts: int,
    device: str,
    removal_counts: List[int] = [1, 2, 4, 6, 8],
) -> Dict:
    """
    Test scaffold-constrained reconstruction for multiple removal counts.
    
    For each removal count (e.g., 1, 2, 4, 6, 8 atoms):
    - Test reconstruction on the same set of molecules
    - Report with_moments vs without_moments rates
    
    Args:
        model: DiT model
        test_smiles: List of test SMILES
        test_scattering: Test scattering moments array
        num_molecules: Number of molecules to test
        attempts: Number of generation attempts per molecule
        device: Device to use
        removal_counts: List of atom counts to remove (tests each separately)
    
    Returns:
        Dict with per-removal-count results and summary
    """
    print(f"\nRunning scaffold reconstruction eval on {num_molecules} molecules...")
    print(f"  Testing removal counts: {removal_counts}")
    print(f"  Attempts per molecule: {attempts}")
    
    # Use same molecules for all removal counts (for fair comparison)
    mol_indices = np.random.choice(len(test_smiles), min(num_molecules, len(test_smiles)), replace=False)
    
    per_removal_results = {}
    
    for num_remove in removal_counts:
        print(f"\n  Testing {num_remove} atom(s) removed...")
        result = scaffold_reconstruction_eval_single(
            model, test_smiles, test_scattering,
            mol_indices, attempts, device, num_remove
        )
        per_removal_results[num_remove] = result
        print(f"    With moments:    {result['with_moments_rate']:.1%} ({result['with_moments_count']}/{result['num_molecules']})")
        print(f"    Without moments: {result['without_moments_rate']:.1%} ({result['without_moments_count']}/{result['num_molecules']})")
    
    return {
        "num_molecules_requested": num_molecules,
        "attempts_per_molecule": attempts,
        "removal_counts": removal_counts,
        "per_removal_results": per_removal_results,
    }


# -----------------------------------------------------------------------------
# Metrics (reused from zinc_eval.py)
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
        return {"validity": 0.0, "validity_filtered": 0.0, "n_valid": 0}
    
    valid_mols = get_valid_mols(smiles_list)
    validity = len(valid_mols) / len(smiles_list)
    
    # Lipinski filter
    filtered_count = 0
    for mol in valid_mols:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
            filtered_count += 1
    
    return {
        "validity": validity,
        "validity_filtered": filtered_count / len(smiles_list),
        "n_valid": len(valid_mols),
    }


def compute_uniqueness(smiles_list: List[str]) -> Dict:
    """Compute uniqueness metrics."""
    valid_smiles = [Chem.MolToSmiles(mol) for mol in get_valid_mols(smiles_list)]
    
    if not valid_smiles:
        return {"uniqueness": 0.0, "unique@1k": 0.0, "unique@10k": 0.0, "n_unique": 0}
    
    unique_smiles = set(valid_smiles)
    return {
        "uniqueness": len(unique_smiles) / len(valid_smiles),
        "unique@1k": len(set(valid_smiles[:1000])) / min(1000, len(valid_smiles)),
        "unique@10k": len(set(valid_smiles[:10000])) / min(10000, len(valid_smiles)),
        "n_unique": len(unique_smiles),
    }


def compute_novelty(smiles_list: List[str], reference_smiles: List[str]) -> Dict:
    """Compute novelty (fraction not in reference set)."""
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


def compute_coverage(smiles_list: List[str]) -> Dict:
    """Compute atom type coverage."""
    found_atoms = set()
    for mol in get_valid_mols(smiles_list):
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol in REFERENCE_ATOM_TYPES:
                found_atoms.add(symbol)
    
    return {
        "coverage": f"{len(found_atoms)}/{len(REFERENCE_ATOM_TYPES)}",
        "coverage_count": len(found_atoms),
        "found_atoms": sorted(list(found_atoms)),
        "missing_atoms": sorted(list(set(REFERENCE_ATOM_TYPES) - found_atoms)),
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


def compute_similarity(smiles_list: List[str], reference_smiles: List[str], sample_size: int = 1000) -> float:
    """Compute nearest-neighbor Tanimoto similarity to reference set."""
    gen_mols = get_valid_mols(smiles_list)
    ref_mols = get_valid_mols(reference_smiles)
    
    if not gen_mols or not ref_mols:
        return 0.0
    
    if len(gen_mols) > sample_size:
        gen_mols = [gen_mols[i] for i in np.random.choice(len(gen_mols), sample_size, replace=False)]
    if len(ref_mols) > sample_size:
        ref_mols = [ref_mols[i] for i in np.random.choice(len(ref_mols), sample_size, replace=False)]
    
    gen_fps = [get_morgan_fingerprint(mol) for mol in gen_mols]
    ref_fps = [get_morgan_fingerprint(mol) for mol in ref_mols]
    
    nn_similarities = []
    for gen_fp in tqdm(gen_fps, desc="Computing similarity", leave=False):
        max_sim = max(DataStructs.TanimotoSimilarity(gen_fp, ref_fp) for ref_fp in ref_fps)
        nn_similarities.append(max_sim)
    
    return float(np.mean(nn_similarities))


def compute_fcd(smiles_list: List[str], reference_smiles: List[str], device: str = "cpu") -> float:
    """Compute Fréchet ChemNet Distance."""
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
    
    print("  Coverage...")
    coverage = compute_coverage(generated_smiles)
    metrics["coverage"] = coverage["coverage"]
    metrics["coverage_details"] = coverage
    
    print("  Diversity...")
    metrics["diversity"] = compute_diversity(generated_smiles)
    
    print("  Similarity...")
    metrics["similarity"] = compute_similarity(generated_smiles, reference_smiles)
    
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
        ("Validity (filtered)", "validity_filtered"),
        ("Uniqueness", "uniqueness"),
        ("Unique@1k", "unique@1k"),
        ("Unique@10k", "unique@10k"),
        ("Novelty", "novelty"),
        ("Coverage", "coverage"),
        ("Diversity", "diversity"),
        ("Similarity", "similarity"),
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
        description="Direct evaluation for GRASSY-DiT on ZINC tranches (no GRASSY VAE needed)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Conditioned generation (default)
  python -m evaluation.zinc_eval_direct \\
      --checkpoint checkpoints/zinc/fbab/checkpoint_best.pt \\
      --test-dir grassy_dit/data/data_fbab/test \\
      --config configs/ZINC/FBAB/FBAB_dit_config.yaml \\
      --num-samples 1000

  # Unconditional generation
  python -m evaluation.zinc_eval_direct \\
      --checkpoint checkpoints/zinc/fbab/checkpoint_best.pt \\
      --test-dir grassy_dit/data/data_fbab/test \\
      --config configs/ZINC/FBAB/FBAB_dit_config.yaml \\
      --num-samples 1000 \\
      --unconditional

  # With reconstruction evaluation
  python -m evaluation.zinc_eval_direct \\
      --checkpoint checkpoints/zinc/fbab/checkpoint_best.pt \\
      --test-dir grassy_dit/data/data_fbab/test \\
      --config configs/ZINC/FBAB/FBAB_dit_config.yaml \\
      --num-samples 1000 \\
      --recon-samples 50 \\
      --recon-attempts 3
        """,
    )
    
    # Required arguments
    parser.add_argument("--checkpoint", required=True, help="Path to DiT model checkpoint")
    parser.add_argument("--test-dir", required=True, help="Directory with test molecules.csv and scattering_moments.npy")
    parser.add_argument("--config", required=True, help="Path to config yaml")
    
    # Generation arguments
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of molecules to generate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--unconditional", action="store_true", help="Generate unconditionally (ignore scattering)")
    
    # Reconstruction evaluation
    parser.add_argument("--recon-samples", type=int, default=0, help="Number of molecules for pure reconstruction eval (0=skip)")
    parser.add_argument("--recon-attempts", type=int, default=3, help="Attempts per molecule in reconstruction eval")
    parser.add_argument("--scaffold-recon-samples", type=int, default=0, help="Number of molecules for scaffold reconstruction eval (0=skip)")
    parser.add_argument("--scaffold-removal-counts", type=str, default="1,2,4,6,8", help="Comma-separated counts of atoms to remove (tests each separately)")
    
    # Output arguments
    parser.add_argument("--output-dir", default=None, help="Output directory (default: evaluation/results/ZINC/)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--smiles-col", default="smiles", help="SMILES column name in CSV")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load test data
    print(f"Loading test data from {args.test_dir}...")
    test_smiles, test_scattering = load_test_data(args.test_dir, args.smiles_col)
    print(f"Loaded {len(test_smiles)} test molecules")
    
    # Load model
    print(f"\nLoading DiT model from {args.checkpoint}...")
    model = load_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        scattering_path=str(Path(args.test_dir) / "scattering_moments.npy"),
    )
    
    # Generate molecules
    if args.unconditional:
        generated_smiles, gen_stats = generate_unconditional(
            model, args.num_samples, args.batch_size, args.device
        )
        mode_str = "unconditional"
    else:
        generated_smiles, gen_stats = generate_conditioned(
            model, test_scattering, args.num_samples, args.batch_size, args.device
        )
        mode_str = "conditioned"
    
    # Compute metrics
    metrics = compute_all_metrics(generated_smiles, test_smiles, args.device)
    
    # Reconstruction evaluation (optional)
    recon_results = None
    if args.recon_samples > 0:
        recon_results = reconstruction_eval(
            model, test_smiles, test_scattering,
            args.recon_samples, args.recon_attempts, args.device
        )
        print(f"\nPure Reconstruction results:")
        print(f"  With moments:    {recon_results['with_moments_rate']:.2%} ({recon_results['with_moments_count']}/{recon_results['num_molecules']})")
        print(f"  Without moments: {recon_results['without_moments_rate']:.2%} ({recon_results['without_moments_count']}/{recon_results['num_molecules']})")
    
    # Scaffold reconstruction evaluation (optional)
    scaffold_recon_results = None
    if args.scaffold_recon_samples > 0:
        removal_counts = [int(x.strip()) for x in args.scaffold_removal_counts.split(",")]
        scaffold_recon_results = scaffold_reconstruction_eval(
            model, test_smiles, test_scattering,
            args.scaffold_recon_samples, args.recon_attempts, args.device,
            removal_counts=removal_counts,
        )
        print(f"\nScaffold Reconstruction Summary:")
        print(f"  {'Atoms Removed':<15} {'With Moments':<20} {'Without Moments':<20}")
        print(f"  {'-'*55}")
        for num_remove in removal_counts:
            r = scaffold_recon_results['per_removal_results'].get(num_remove, {})
            with_rate = r.get('with_moments_rate', 0)
            without_rate = r.get('without_moments_rate', 0)
            with_count = r.get('with_moments_count', 0)
            without_count = r.get('without_moments_count', 0)
            total = r.get('num_molecules', 0)
            print(f"  {num_remove:<15} {with_rate:.1%} ({with_count}/{total})      {without_rate:.1%} ({without_count}/{total})")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Results ({mode_str})")
    print(f"{'='*60}")
    print(format_results_table(metrics))
    
    # Save generated samples
    samples_path = output_dir / f"generated_{mode_str}_{timestamp}.txt"
    with open(samples_path, "w") as f:
        for smi in generated_smiles:
            f.write(smi + "\n")
    print(f"\nSaved samples to {samples_path}")
    
    # Save results
    results = {
        "config": vars(args),
        "generation_stats": gen_stats,
        "metrics": metrics,
        "reconstruction": recon_results,
        "scaffold_reconstruction": scaffold_recon_results,
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
    
    results_path = output_dir / f"zinc_eval_{mode_str}_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(convert_to_native(results), f, indent=2)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
