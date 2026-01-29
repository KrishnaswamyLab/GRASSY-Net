"""
Stage 6: Evaluation

Generates molecules and computes evaluation metrics.
Wraps evaluation/zinc_eval.py functionality.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

import numpy as np
import yaml
import torch
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Lipinski

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grassy_dit.train import ScatteringGraphDIT
from models.GRASSY_model import GRASSY

# Try to import FCD
try:
    from fcd_torch import FCD as FCDMetric
    HAS_FCD = True
except ImportError:
    HAS_FCD = False
    print("Warning: fcd_torch not installed, FCD metric will be skipped")

REFERENCE_ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I"]


def load_dit_model(checkpoint_path: str, config_path: str, device: str = "cpu"):
    """Load DiT model from checkpoint."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = ScatteringGraphDIT(config)
    model.device = torch.device(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    scattering_cfg = config.get('scattering', {})
    J = scattering_cfg.get('J', 4)
    num_levels = 1 + J + J * (J - 1) // 2
    num_moments = scattering_cfg.get('num_moments', 4)
    
    state = checkpoint.get("model_state_dict", {})
    level_proj_w = state.get("denoiser.scatter_tokenizer.level_proj.weight")
    
    if level_proj_w is not None:
        model.num_atom_types = level_proj_w.shape[1] // num_moments
    else:
        model.num_atom_types = 16  # Default
    
    model.num_levels = num_levels
    model.num_moments = num_moments
    model.J = J
    
    model._initialize_model(model.model_class, checkpoint)
    model.is_fitted_ = True
    
    return model


def load_grassy_model(checkpoint_path: str, device: str = "cpu"):
    """Load GRASSY autoencoder from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        model = GRASSY.load_from_checkpoint(checkpoint_path, hparams=hparams, map_location=device)
    else:
        raise ValueError("No hyper_parameters found in checkpoint")
    
    model.eval()
    model.to(device)
    return model


def load_smiles_from_dir(data_dir: str) -> List[str]:
    """Load SMILES from a data directory."""
    import pandas as pd
    csv_path = os.path.join(data_dir, 'molecules.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df['smiles'].tolist()
    return []


def canonicalize(smiles: str) -> Optional[str]:
    """Canonicalize a SMILES string."""
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


def compute_validity(smiles_list: List[str]) -> Dict[str, float]:
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


def compute_uniqueness(smiles_list: List[str]) -> Dict[str, float]:
    """Compute uniqueness metrics."""
    valid_smiles = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            valid_smiles.append(Chem.MolToSmiles(mol))
    
    if not valid_smiles:
        return {"uniqueness": 0.0, "n_unique": 0}
    
    unique_smiles = set(valid_smiles)
    uniqueness = len(unique_smiles) / len(valid_smiles)
    
    return {
        "uniqueness": uniqueness,
        "n_unique": len(unique_smiles),
    }


def compute_novelty(smiles_list: List[str], reference_smiles: List[str]) -> Dict[str, float]:
    """Compute novelty."""
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


def compute_diversity(smiles_list: List[str], sample_size: int = 5000) -> float:
    """Compute internal diversity."""
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


def compute_fcd(smiles_list: List[str], reference_smiles: List[str], device: str = "cpu") -> float:
    """Compute FCD if available."""
    if not HAS_FCD:
        return float('nan')
    
    gen_smiles = [Chem.MolToSmiles(mol) for mol in get_valid_mols(smiles_list)]
    ref_smiles = [Chem.MolToSmiles(mol) for mol in get_valid_mols(reference_smiles)]
    
    if not gen_smiles:
        return float('inf')
    
    fcd_metric = FCDMetric(device=device)
    return float(fcd_metric(gen=gen_smiles, ref=ref_smiles))


def run_evaluate(
    test_dir: str,
    train_dir: str,
    dit_checkpoint: str,
    dit_config: str,
    output_dir: str,
    grassy_checkpoint: Optional[str] = None,
    num_samples: int = 1000,
    batch_size: int = 64,
    guide_scale: float = 2.0,
    device: str = "auto",
) -> Tuple[str, Dict[str, Any]]:
    """
    Run evaluation stage.
    
    Args:
        test_dir: Directory containing test data
        train_dir: Directory containing training data (for novelty computation)
        dit_checkpoint: Path to DiT checkpoint
        dit_config: Path to DiT config
        output_dir: Directory to save results
        grassy_checkpoint: Optional path to GRASSY checkpoint (for latent sampling)
        num_samples: Number of molecules to generate
        batch_size: Generation batch size
        guide_scale: Classifier-free guidance scale
        device: Device to use
    
    Returns:
        Tuple of (report_path, metrics_dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Stage 6: Evaluation")
    print("=" * 60)
    
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load test data
    print(f"\nLoading test data from: {test_dir}")
    test_smiles = load_smiles_from_dir(test_dir)
    test_scattering = np.load(os.path.join(test_dir, 'scattering_moments.npy'))
    print(f"Loaded {len(test_smiles)} test molecules")
    
    # Load training data for novelty
    print(f"Loading training data from: {train_dir}")
    train_smiles = load_smiles_from_dir(train_dir)
    print(f"Loaded {len(train_smiles)} training molecules")
    
    # Load DiT model
    print(f"\nLoading DiT model from: {dit_checkpoint}")
    dit_model = load_dit_model(dit_checkpoint, dit_config, device)
    
    # Generate molecules conditioned on test scattering moments
    print(f"\nGenerating {num_samples} molecules...")
    all_generated = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_samples)
        current_batch_size = batch_end - batch_start
        
        # Sample random scattering moments from test set
        indices = np.random.choice(len(test_scattering), current_batch_size, replace=True)
        batch_scattering = torch.from_numpy(test_scattering[indices]).float()
        
        try:
            generated = dit_model.generate(
                scattering=batch_scattering,
                batch_size=current_batch_size,
            )
            
            for smi in generated:
                if smi is not None:
                    all_generated.append(smi)
        except Exception as e:
            print(f"Warning: Generation failed for batch {batch_idx}: {e}")
            continue
    
    print(f"Generated {len(all_generated)} valid SMILES")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = {}
    
    print("  Validity...")
    metrics.update(compute_validity(all_generated))
    
    print("  Uniqueness...")
    metrics.update(compute_uniqueness(all_generated))
    
    print("  Novelty...")
    metrics.update(compute_novelty(all_generated, train_smiles))
    
    print("  Diversity...")
    metrics["diversity"] = compute_diversity(all_generated)
    
    print("  FCD...")
    metrics["fcd"] = compute_fcd(all_generated, test_smiles, device)
    
    # Save generated molecules
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_path = os.path.join(output_dir, f"generated_samples_{timestamp}.txt")
    with open(samples_path, 'w') as f:
        for smi in all_generated:
            f.write(smi + "\n")
    print(f"\nSaved samples to: {samples_path}")
    
    # Save metrics
    results = {
        'config': {
            'test_dir': test_dir,
            'train_dir': train_dir,
            'dit_checkpoint': dit_checkpoint,
            'num_samples': num_samples,
            'batch_size': batch_size,
            'guide_scale': guide_scale,
        },
        'metrics': metrics,
        'timestamp': timestamp,
    }
    
    report_path = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results to: {report_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    print(f"  Validity:          {metrics.get('validity', 0):.4f}")
    print(f"  Validity (Lipinski): {metrics.get('validity_filtered', 0):.4f}")
    print(f"  Uniqueness:        {metrics.get('uniqueness', 0):.4f}")
    print(f"  Novelty:           {metrics.get('novelty', 0):.4f}")
    print(f"  Diversity:         {metrics.get('diversity', 0):.4f}")
    print(f"  FCD:               {metrics.get('fcd', float('nan')):.4f}")
    
    print(f"\nEvaluation complete!")
    return report_path, metrics
