"""
Evaluation script for GRASSY-DiT molecular generation with latent space sampling.

Generates molecules by:
1. Randomly sampling from the GRASSY autoencoder's latent space (standard normal prior)
2. Decoding latent vectors to scattering moments
3. Using scattering moments to condition DiT generation

Metrics: Validity, Uniqueness, Novelty, Coverage, Diversity, Similarity, FCD.

Usage:
    python -m evaluation.zinc_eval_latent \
        --dataset grassy_dit/data/data_bbab/test \
        --dit-checkpoint checkpoints/zinc/bbab/checkpoint_best.pt \
        --grassy-checkpoint-dir outputs/GRASSY_model/ \
        --config grassy_dit/grassy_dit_config.yaml \
        --output-dir evaluation/results/ZINC/latent_sampling \
        --num-samples 1000

    # Basic latent sampling evaluation
    python -m evaluation.zinc_eval \
        --dataset grassy_dit/data/data_bbab/test \
        --dit-checkpoint checkpoints/zinc/bbab/checkpoint_best.pt \
        --grassy-checkpoint-dir outputs/BBAB_fixed_2026-01-26-20-30-17 \
        --config checkpoints/zinc/bbab/config_2026-01-26-21-43-08.yaml \
        --output-dir evaluation/results/ZINC/latent_sampling \
        --num-samples 10
"""

import argparse
import json
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import yaml
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski
from tqdm import tqdm


from fcd_torch import FCD as FCDMetric

# Import model loaders from your existing code
from grassy_dit.train import ScatteringGraphDIT
from models.GRASSY_model import GRASSY

# Reference heavy atom types (common in drug-like molecules)
REFERENCE_ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I"]


# -----------------------------------------------------------------------------
# Model Loading (adapted from sample_target_optimization.py)
# -----------------------------------------------------------------------------

def load_dit_model(checkpoint_path: str, config_path: str, device: str = "cpu", scattering_path: str = None):
    """Load GRASSY-DiT model from checkpoint."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = ScatteringGraphDIT(config)
    model.device = torch.device(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    scattering_cfg = config.get('scattering', {})
    J = scattering_cfg.get('J', 4)
    num_levels = 1 + J + J * (J - 1) // 2
    num_moments = scattering_cfg.get('num_moments', 4)

    if scattering_path is not None:
        scattering_data = np.load(scattering_path)
        model.num_atom_types = scattering_data.shape[-1] // (num_levels * num_moments)
    else:
        state = checkpoint.get("model_state_dict", {})
        level_proj_w = state.get("denoiser.scatter_tokenizer.level_proj.weight")
        pos = state.get("denoiser.scatter_tokenizer.pos")

        if level_proj_w is not None:
            model.num_atom_types = level_proj_w.shape[1] // num_moments
        elif pos is not None:
            model.num_atom_types = pos.shape[1] - num_levels

    model.num_levels = num_levels
    model.num_moments = num_moments
    model.J = J

    model._initialize_model(model.model_class, checkpoint)
    model.is_fitted_ = True
    model.fitting_loss = [0.0]
    model.fitting_epoch = 0

    hparams = checkpoint.get('hyperparameters', {})
    if not getattr(model, "dataset_info", None):
        model.dataset_info = hparams.get("dataset_info", None)

    return model


def load_grassy_model(checkpoint_dir: str, device: str = "cpu") -> Tuple[GRASSY, dict]:
    """Load the GRASSY autoencoder model with its config."""
    config_path = Path(checkpoint_dir) / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint_files = glob(str(Path(checkpoint_dir) / 'best-epoch=*.ckpt'))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")
    checkpoint_path = checkpoint_files[0]
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        model = GRASSY.load_from_checkpoint(checkpoint_path, hparams=hparams, map_location=device)
    else:
        raise ValueError("No hyper_parameters found in checkpoint")
    
    model.eval()
    model.to(device)
    return model, config


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_smiles_from_dataset(dataset_path: str, smiles_column: str = "smiles") -> List[str]:
    """Load SMILES from a dataset path (.csv, .npy, .txt, .smi, or directory)."""
    path = Path(dataset_path)
    all_smiles = []

    if path.is_dir():
        csv_file = path / "molecules.csv"
        if csv_file.exists():
            return _load_csv_file(csv_file, smiles_column)
        
        for ext in ["*.csv", "*.npy", "*.txt", "*.smi"]:
            for file in sorted(path.glob(ext)):
                if "_stats" in file.name:
                    continue
                all_smiles.extend(_load_single_file(file, smiles_column))
    else:
        all_smiles = _load_single_file(path, smiles_column)

    return all_smiles


def _load_csv_file(file_path: Path, smiles_column: str = "smiles") -> List[str]:
    """Load SMILES from a CSV file."""
    import csv
    smiles = []
    with open(file_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            smi = row.get(smiles_column, "").strip()
            if smi:
                smiles.append(smi)
    return smiles


def _load_single_file(file_path: Path, smiles_column: str = "smiles") -> List[str]:
    """Load SMILES from a single file."""
    if file_path.suffix == ".csv":
        return _load_csv_file(file_path, smiles_column)
    elif file_path.suffix == ".npy":
        data = np.load(file_path, allow_pickle=True).item()
        return list(data.keys())
    elif file_path.suffix in [".txt", ".smi"]:
        with open(file_path) as f:
            return [line.strip() for line in f if line.strip()]
    return []


# -----------------------------------------------------------------------------
# Latent Space Sampling & Generation
# -----------------------------------------------------------------------------

def sample_latent_space(
    grassy_model: GRASSY,
    num_samples: int,
    device: str = "cpu",
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Sample from the GRASSY latent space (standard normal prior).
    
    Args:
        grassy_model: Trained GRASSY autoencoder
        num_samples: Number of latent vectors to sample
        device: Device to use
        temperature: Sampling temperature (scales the standard deviation)
    
    Returns:
        Tensor of shape [num_samples, latent_dim]
    """
    latent_dim = grassy_model.hparams.get('bottle_dim', grassy_model.bottle_dim)
    
    # Sample from standard normal prior (VAE assumption)
    z = torch.randn(num_samples, latent_dim, device=device) * temperature
    
    return z


def decode_latent_to_scattering(
    grassy_model: GRASSY,
    z: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Decode latent vectors to scattering moments and predicted num_atoms.
    
    GRASSY model structure:
    - decode(z) -> reconstructed scattering moments
    - predict(z) -> (y_full, y_pred, num_atoms_pred) where num_atoms is last column of y_full
    
    Args:
        grassy_model: Trained GRASSY autoencoder
        z: Latent vectors [batch_size, bottle_dim]
    
    Returns:
        scattering: Decoded scattering moments [batch_size, input_dim]
        num_atoms: Predicted number of atoms [batch_size]
    """
    with torch.no_grad():
        # Decode latent to scattering moments
        scattering = grassy_model.decode(z)
        
        # Get num_atoms prediction from the predict head
        # predict() returns (y_full, y_pred, num_atoms_pred)
        # y_full[:, -1] is num_atoms (last property)
        y_full, _, num_atoms_pred = grassy_model.predict(z)
        
        # num_atoms_pred is [batch_size, 1], squeeze to [batch_size]
        num_atoms = num_atoms_pred.squeeze(-1)
    
    return scattering, num_atoms


def generate_from_latent(
    dit_model,
    grassy_model: GRASSY,
    num_samples: int,
    batch_size: int,
    device: str,
    temperature: float = 1.0,
    use_predicted_num_atoms: bool = True,
    fixed_num_atoms: Optional[int] = None,
) -> Tuple[List[str], dict]:
    """
    Generate molecules by sampling from GRASSY latent space.
    
    Args:
        dit_model: GRASSY-DiT generation model
        grassy_model: GRASSY autoencoder for latent sampling
        num_samples: Total number of molecules to generate
        batch_size: Batch size for generation
        device: Device to use
        temperature: Latent sampling temperature
        use_predicted_num_atoms: Whether to use GRASSY's predicted num_atoms
        fixed_num_atoms: If set, use this fixed number of atoms for all molecules
    
    Returns:
        all_smiles: List of generated SMILES
        stats: Generation statistics
    """
    print(f"\nGenerating {num_samples} molecules from latent space...")
    print(f"  Temperature: {temperature}")
    print(f"  Use predicted num_atoms: {use_predicted_num_atoms}")
    
    all_smiles = []
    all_num_atoms = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_samples)
        current_batch_size = batch_end - batch_start
        
        # Sample from latent space
        z = sample_latent_space(grassy_model, current_batch_size, device, temperature)
        
        # Decode to scattering moments
        scattering, num_atoms_pred = decode_latent_to_scattering(grassy_model, z)
        
       # Determine num_nodes for whole batch
        if fixed_num_atoms is not None:
            num_nodes = torch.full((current_batch_size,), fixed_num_atoms, dtype=torch.long)
        elif use_predicted_num_atoms:
            num_nodes = torch.clamp(torch.round(num_atoms_pred).long(), min=1)
        else:
            num_nodes = None

        # Track num_atoms for stats
        if num_nodes is not None:
            all_num_atoms.extend(num_nodes.cpu().tolist())
        else:
            all_num_atoms.extend([-1] * current_batch_size)
        
        # Generate molecules in batch - DiT handles batched scattering tensors
        try:
            smiles_list = dit_model.generate(
                scattering=scattering,  # [batch_size, scattering_dim] tensor on device
                num_nodes=num_nodes,    # [batch_size] tensor or None
                batch_size=current_batch_size,
            )
            
            # Filter valid SMILES
            for smi in smiles_list:
                if smi is not None:
                    all_smiles.append(smi)
                    
        except Exception as e:
            print(f"Warning: Batch generation failed for batch {batch_idx}: {e}")
            continue
    
    stats = {
        "num_generated": len(all_smiles),
        "num_attempted": num_samples,
        "success_rate": len(all_smiles) / num_samples if num_samples > 0 else 0,
        "avg_num_atoms": np.mean([n for n in all_num_atoms if n > 0]) if all_num_atoms else 0,
        "temperature": temperature,
    }
    
    print(f"Generated {len(all_smiles)} valid molecules ({stats['success_rate']:.1%} success rate)")
    
    return all_smiles, stats


def generate_from_reference_scattering(
    dit_model,
    grassy_model: GRASSY,
    reference_smiles: List[str],
    num_samples: int,
    batch_size: int,
    device: str,
    scattering_model=None,
    atom_types: List[str] = None,
) -> Tuple[List[str], dict]:
    """
    Alternative: Generate by encoding reference molecules then sampling around them.
    
    This provides a "reconstruction + variation" evaluation mode.
    """
    from evaluation.utils import compute_scattering_from_smiles
    from models.ScatteringTransform import GraphScatteringTransform
    
    print(f"\nGenerating {num_samples} molecules from reference scattering...")
    
    # Setup scattering model if not provided
    if scattering_model is None:
        scattering_model = GraphScatteringTransform(
            in_channels=len(atom_types) if atom_types else 10,
            J=dit_model.J,
            num_moments=dit_model.num_moments,
        ).to(device)
        scattering_model.eval()
    
    if atom_types is None:
        atom_types = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"]
    
    all_smiles = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating from reference"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_samples)
        current_batch_size = batch_end - batch_start
        
        # Sample random reference molecules
        indices = np.random.choice(len(reference_smiles), current_batch_size, replace=True)
        batch_smiles = [reference_smiles[i] for i in indices]
        
        # Compute scattering for reference molecules
        scattering = compute_scattering_from_smiles(batch_smiles, scattering_model, atom_types, device)
        
        # Encode into latent space
        with torch.no_grad():
            z, mu, logvar = grassy_model.embed(scattering.to(device))
        
        # Add small noise for variation (optional)
        # z = z + torch.randn_like(z) * 0.1
        
        # Decode back to scattering
        scattering_decoded, num_atoms_pred = decode_latent_to_scattering(grassy_model, z)
        
        # Generate molecules
        for i in range(current_batch_size):
            scattering_np = scattering_decoded[i].cpu().numpy()
            num_nodes = max(1, int(round(num_atoms_pred[i].item())))
            
            try:
                smiles_list = dit_model.generate(
                    scattering=scattering_np,
                    num_nodes=num_nodes,
                    batch_size=1,
                )
                
                if smiles_list and smiles_list[0] is not None:
                    all_smiles.append(smiles_list[0])
            except Exception as e:
                continue
    
    stats = {
        "num_generated": len(all_smiles),
        "num_attempted": num_samples,
        "success_rate": len(all_smiles) / num_samples if num_samples > 0 else 0,
        "mode": "reference_scattering",
    }
    
    return all_smiles, stats


# -----------------------------------------------------------------------------
# Metrics (same as zinc_eval.py)
# -----------------------------------------------------------------------------

def canonicalize(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string, return None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None


def get_valid_mols(smiles_list: List[str]) -> List:
    """Convert SMILES to valid RDKit mol objects."""
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mols.append(mol)
    return mols


def get_morgan_fingerprint(mol, radius: int = 2, n_bits: int = 2048):
    """Get Morgan fingerprint for a molecule."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def compute_validity(smiles_list: List[str]) -> dict:
    """Compute validity metrics."""
    if not smiles_list:
        return {"validity": 0.0, "validity_filtered": 0.0, "n_valid": 0}

    valid_mols = get_valid_mols(smiles_list)
    validity = len(valid_mols) / len(smiles_list)

    filtered_count = 0
    for mol in valid_mols:
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        if mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10:
            filtered_count += 1

    validity_filtered = filtered_count / len(smiles_list) if smiles_list else 0.0

    return {
        "validity": validity,
        "validity_filtered": validity_filtered,
        "n_valid": len(valid_mols),
    }


def compute_uniqueness(smiles_list: List[str]) -> dict:
    """Compute uniqueness metrics."""
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            valid_smiles.append(Chem.MolToSmiles(mol))

    if not valid_smiles:
        return {"uniqueness": 0.0, "unique@1k": 0.0, "unique@10k": 0.0, "n_unique": 0}

    unique_smiles = set(valid_smiles)
    uniqueness = len(unique_smiles) / len(valid_smiles)

    unique_1k = len(set(valid_smiles[:1000])) / min(1000, len(valid_smiles))
    unique_10k = len(set(valid_smiles[:10000])) / min(10000, len(valid_smiles))

    return {
        "uniqueness": uniqueness,
        "unique@1k": unique_1k,
        "unique@10k": unique_10k,
        "n_unique": len(unique_smiles),
    }


def compute_novelty(smiles_list: List[str], reference_smiles: List[str]) -> dict:
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

    novelty = novel_count / valid_count if valid_count > 0 else 0.0
    return {"novelty": novelty, "n_novel": novel_count}


def compute_coverage(smiles_list: List[str]) -> dict:
    """Compute atom type coverage."""
    found_atoms = set()

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
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

    n = len(fps)
    total_distance = 0.0
    count = 0

    for i in range(n):
        for j in range(i + 1, n):
            similarity = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            total_distance += 1 - similarity
            count += 1

    return total_distance / count if count > 0 else 0.0


def compute_similarity(
    smiles_list: List[str],
    reference_smiles: List[str],
    sample_size: int = 1000,
) -> float:
    """Compute nearest-neighbor Tanimoto similarity to reference set."""
    gen_mols = get_valid_mols(smiles_list)
    ref_mols = get_valid_mols(reference_smiles)

    if not gen_mols or not ref_mols:
        return 0.0

    if len(gen_mols) > sample_size:
        indices = np.random.choice(len(gen_mols), sample_size, replace=False)
        gen_mols = [gen_mols[i] for i in indices]

    if len(ref_mols) > sample_size:
        indices = np.random.choice(len(ref_mols), sample_size, replace=False)
        ref_mols = [ref_mols[i] for i in indices]

    gen_fps = [get_morgan_fingerprint(mol) for mol in gen_mols]
    ref_fps = [get_morgan_fingerprint(mol) for mol in ref_mols]

    nn_similarities = []
    for gen_fp in tqdm(gen_fps, desc="Computing similarity", leave=False):
        max_sim = max(DataStructs.TanimotoSimilarity(gen_fp, ref_fp) for ref_fp in ref_fps)
        nn_similarities.append(max_sim)

    return float(np.mean(nn_similarities))


def compute_fcd(
    smiles_list: List[str],
    reference_smiles: List[str],
    device: str = "cpu",
) -> float:
    """Compute Fréchet ChemNet Distance."""
    gen_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            gen_smiles.append(Chem.MolToSmiles(mol))

    ref_smiles = []
    for smi in reference_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            ref_smiles.append(Chem.MolToSmiles(mol))

    if not gen_smiles:
        return float("inf")

    fcd_metric = FCDMetric(device=device)
    return float(fcd_metric(gen=gen_smiles, ref=ref_smiles))


def compute_all_metrics(
    generated_smiles: List[str],
    reference_smiles: List[str],
    device: str = "cpu",
) -> dict:
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

def format_results_table(metrics: dict) -> str:
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
        if value is None:
            rows.append(f"| {display_name} | N/A |")
        elif isinstance(value, float):
            if np.isnan(value):
                rows.append(f"| {display_name} | N/A |")
            else:
                rows.append(f"| {display_name} | {value:.4f} |")
        else:
            rows.append(f"| {display_name} | {value} |")

    return "\n".join(rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluation for GRASSY-DiT with latent space sampling",
    )
    # Data arguments
    parser.add_argument("--dataset", required=True, help="Path to reference dataset")
    
    # Model arguments
    parser.add_argument("--dit-checkpoint", required=True, help="DiT model checkpoint")
    parser.add_argument("--grassy-checkpoint-dir", required=True, help="GRASSY autoencoder checkpoint directory")
    parser.add_argument("--config", default="grassy_dit/grassy_dit_config.yaml", help="DiT config path")
    parser.add_argument("--scattering-path", default=None, help="Optional: precomputed scattering .npy for dimension inference")
    
    # Generation arguments
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for generation")
    parser.add_argument("--temperature", type=float, default=1.0, help="Latent sampling temperature")
    parser.add_argument("--use-predicted-num-atoms", action="store_true", default=True,
                        help="Use GRASSY's predicted num_atoms (default: True)")
    parser.add_argument("--fixed-num-atoms", type=int, default=None,
                        help="Fixed number of atoms for all molecules (overrides predicted)")
    
    # Generation mode
    parser.add_argument("--mode", choices=["latent", "reference"], default="latent",
                        help="Generation mode: 'latent' samples from prior, 'reference' encodes reference molecules")
    
    # Output arguments
    parser.add_argument("--output-dir", default="./eval_results", help="Output directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load reference data
    print(f"Loading dataset from {args.dataset}...")
    reference_smiles = load_smiles_from_dataset(args.dataset)
    print(f"Loaded {len(reference_smiles)} reference molecules")

    # Load models
    print(f"\nLoading DiT model from {args.dit_checkpoint}...")
    dit_model = load_dit_model(
        args.dit_checkpoint, 
        args.config, 
        args.device,
        scattering_path=args.scattering_path
    )

    print(f"\nLoading GRASSY model from {args.grassy_checkpoint_dir}...")
    grassy_model, grassy_config = load_grassy_model(args.grassy_checkpoint_dir, args.device)
    
    # Print model info
    latent_dim = grassy_model.hparams.get('latent_dim', getattr(grassy_model, 'latent_dim', 'unknown'))
    print(f"  GRASSY latent dimension: {latent_dim}")

    # Generate molecules
    print(f"\n{'='*60}")
    print(f"Generation Mode: {args.mode}")
    print(f"{'='*60}")

    if args.mode == "latent":
        generated_smiles, gen_stats = generate_from_latent(
            dit_model=dit_model,
            grassy_model=grassy_model,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=args.device,
            temperature=args.temperature,
            use_predicted_num_atoms=args.use_predicted_num_atoms,
            fixed_num_atoms=args.fixed_num_atoms,
        )
    else:  # reference mode
        generated_smiles, gen_stats = generate_from_reference_scattering(
            dit_model=dit_model,
            grassy_model=grassy_model,
            reference_smiles=reference_smiles,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=args.device,
        )

    # Save generated samples
    samples_path = output_dir / f"generated_latent_{timestamp}.txt"
    with open(samples_path, "w") as f:
        f.write(f"# Mode: {args.mode}\n")
        f.write(f"# Temperature: {args.temperature}\n")
        f.write(f"# Num samples: {args.num_samples}\n")
        f.write(f"# Generated: {len(generated_smiles)}\n")
        f.write("#\n")
        for smi in generated_smiles:
            f.write(smi + "\n")
    print(f"\nSaved samples to {samples_path}")

    # Compute metrics
    metrics = compute_all_metrics(generated_smiles, reference_smiles, args.device)
    
    # Add generation stats to metrics
    metrics["generation"] = gen_stats
    
    print(f"\n{'='*60}")
    print("Results:")
    print(f"{'='*60}")
    print(format_results_table(metrics))

    # Save results
    results = {
        "config": vars(args),
        "generation_stats": gen_stats,
        "metrics": metrics,
        "timestamp": timestamp,
    }
    
    results_path = output_dir / f"eval_results_latent_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved results to {results_path}")


if __name__ == "__main__":
    main()