"""
Stage 7: Unconstrained Sampling

Generates molecules by sampling from the prior distribution in GRASSY's latent space.
Wraps grassy_dit/sample_unconstrained.py functionality.

Computes metrics for unconditional generation evaluation:
- Validity (Val): % of generated SMILES that are valid molecules
- Uniqueness (Uniq): % of valid molecules that are unique
- Novelty (Nov): % of unique molecules not in training set
- Diversity (Div): Average pairwise Tanimoto distance among generated molecules
- Similarity (Sim): Average max Tanimoto similarity to training set
- FCD: Fréchet ChemNet Distance to training set
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Set
from datetime import datetime

import numpy as np
import torch
import yaml
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try to import FCD
try:
    from fcd_torch import FCD as FCDMetric
    HAS_FCD = True
except ImportError:
    HAS_FCD = False

from grassy_dit.train import ScatteringGraphDIT
from models.GRASSY_model import GRASSY


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
        model.num_atom_types = 16
    
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


def get_morgan_fingerprint(mol, radius: int = 2, n_bits: int = 2048):
    """Get Morgan fingerprint for a molecule."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def compute_metrics(
    generated_smiles: List[str],
    training_smiles: Optional[Set[str]] = None,
    training_smiles_list: Optional[List[str]] = None,
    device: str = "cpu",
) -> Dict[str, float]:
    """Compute all unconditional generation metrics."""
    
    n_total = len(generated_smiles)
    
    # Validity
    valid_mols = []
    valid_smiles = []
    for smi in generated_smiles:
        if smi is None:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            valid_mols.append(mol)
            valid_smiles.append(Chem.MolToSmiles(mol))
    
    n_valid = len(valid_mols)
    validity = (n_valid / n_total * 100) if n_total > 0 else 0
    
    # Uniqueness
    unique_smiles = list(set(valid_smiles))
    n_unique = len(unique_smiles)
    uniqueness = (n_unique / n_valid * 100) if n_valid > 0 else 0
    
    # Novelty (if training set provided)
    novelty = 0.0
    n_novel = 0
    if training_smiles is not None:
        novel_smiles = [s for s in unique_smiles if s not in training_smiles]
        n_novel = len(novel_smiles)
        novelty = (n_novel / n_unique * 100) if n_unique > 0 else 0
    
    # Diversity (avg pairwise Tanimoto distance)
    diversity = 0.0
    if len(valid_mols) >= 2:
        fps = [get_morgan_fingerprint(mol) for mol in valid_mols[:5000]]  # Cap for speed
        distances = []
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                distances.append(1 - sim)
        diversity = (np.mean(distances) * 100) if distances else 0
    
    # Similarity to training set
    similarity = 0.0
    if training_smiles_list is not None and len(valid_mols) > 0:
        train_mols = [Chem.MolFromSmiles(s) for s in training_smiles_list[:5000]]
        train_mols = [m for m in train_mols if m is not None]
        
        if train_mols:
            train_fps = [get_morgan_fingerprint(mol) for mol in train_mols]
            gen_fps = [get_morgan_fingerprint(mol) for mol in valid_mols[:1000]]
            
            max_sims = []
            for gen_fp in gen_fps:
                sims = [DataStructs.TanimotoSimilarity(gen_fp, train_fp) for train_fp in train_fps]
                max_sims.append(max(sims))
            similarity = np.mean(max_sims) * 100
    
    # FCD
    fcd = float('nan')
    if HAS_FCD and training_smiles_list is not None and len(valid_smiles) > 0:
        try:
            fcd_metric = FCDMetric(device=device)
            fcd = float(fcd_metric(gen=valid_smiles, ref=training_smiles_list[:10000]))
        except Exception as e:
            print(f"FCD computation failed: {e}")
    
    return {
        'n_total': n_total,
        'n_valid': n_valid,
        'n_unique': n_unique,
        'n_novel': n_novel,
        'validity': validity,
        'uniqueness': uniqueness,
        'novelty': novelty,
        'diversity': diversity,
        'similarity': similarity,
        'fcd': fcd,
    }


def run_sample_unconstrained(
    dit_checkpoint: str,
    dit_config: str,
    grassy_checkpoint: str,
    output_dir: str,
    training_smiles_path: Optional[str] = None,
    n_samples: int = 1000,
    batch_size: int = 32,
    sampling_method: str = "prior",
    device: str = "auto",
) -> Tuple[str, Dict[str, Any]]:
    """
    Run unconstrained sampling stage.
    
    Args:
        dit_checkpoint: Path to DiT checkpoint
        dit_config: Path to DiT config
        grassy_checkpoint: Path to GRASSY checkpoint
        output_dir: Directory to save results
        training_smiles_path: Path to training SMILES file (for novelty/similarity/FCD)
        n_samples: Number of molecules to generate
        batch_size: Generation batch size
        sampling_method: Latent sampling method (prior, training, noisy_training)
        device: Device to use
    
    Returns:
        Tuple of (samples_path, metrics_dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Stage 7: Unconstrained Sampling")
    print("=" * 60)
    
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load models
    print(f"\nLoading DiT model from: {dit_checkpoint}")
    dit_model = load_dit_model(dit_checkpoint, dit_config, device)
    
    print(f"Loading GRASSY model from: {grassy_checkpoint}")
    grassy_model = load_grassy_model(grassy_checkpoint, device)
    
    # Load training SMILES if provided
    training_smiles_set = None
    training_smiles_list = None
    if training_smiles_path and os.path.exists(training_smiles_path):
        print(f"Loading training SMILES from: {training_smiles_path}")
        with open(training_smiles_path, 'r') as f:
            training_smiles_list = [line.strip().split()[0] for line in f 
                                   if line.strip() and not line.startswith('#')]
        # Canonicalize for novelty check
        training_smiles_set = set()
        for smi in training_smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                training_smiles_set.add(Chem.MolToSmiles(mol))
        print(f"  Loaded {len(training_smiles_set)} training molecules")
    
    # Sample from prior
    print(f"\nSampling {n_samples} latent vectors from prior...")
    latent_dim = grassy_model.hparams.get('bottle_dim', 32)
    
    all_generated = []
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="Generating"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_samples)
        current_batch_size = batch_end - batch_start
        
        # Sample from prior N(0, I)
        z = torch.randn(current_batch_size, latent_dim, device=device)
        
        # Decode to scattering moments
        with torch.no_grad():
            y_pred, scattering, num_atoms = grassy_model.predict(z)
        
        # Generate molecules
        try:
            for i in range(current_batch_size):
                scatter_i = scattering[i].cpu().numpy()
                num_nodes = int(round(num_atoms[i].item()))
                num_nodes = max(5, min(50, num_nodes))  # Clamp
                
                generated = dit_model.generate(
                    scattering=scatter_i,
                    num_nodes=num_nodes,
                    batch_size=1,
                )
                
                if generated and generated[0]:
                    all_generated.append(generated[0])
        except Exception as e:
            print(f"Warning: Generation failed for batch {batch_idx}: {e}")
            continue
    
    print(f"\nGenerated {len(all_generated)} SMILES")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(
        all_generated,
        training_smiles_set,
        training_smiles_list,
        device,
    )
    
    # Save generated molecules
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_path = os.path.join(output_dir, f"unconstrained_samples_{timestamp}.txt")
    with open(samples_path, 'w') as f:
        f.write(f"# Unconstrained generation from prior\n")
        f.write(f"# n_samples: {n_samples}\n")
        f.write(f"# sampling_method: {sampling_method}\n")
        for smi in all_generated:
            if smi:
                f.write(smi + "\n")
    print(f"Saved samples to: {samples_path}")
    
    # Save metrics
    results = {
        'config': {
            'dit_checkpoint': dit_checkpoint,
            'grassy_checkpoint': grassy_checkpoint,
            'n_samples': n_samples,
            'sampling_method': sampling_method,
        },
        'metrics': metrics,
        'timestamp': timestamp,
    }
    
    metrics_path = os.path.join(output_dir, f"unconstrained_metrics_{timestamp}.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved metrics to: {metrics_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Unconstrained Generation Results:")
    print("=" * 60)
    print(f"  Validity:      {metrics['validity']:.2f}%")
    print(f"  Uniqueness:    {metrics['uniqueness']:.2f}%")
    print(f"  Novelty:       {metrics['novelty']:.2f}%")
    print(f"  Diversity:     {metrics['diversity']:.2f}%")
    print(f"  Similarity:    {metrics['similarity']:.2f}%")
    fcd_str = f"{metrics['fcd']:.4f}" if not np.isnan(metrics['fcd']) else "N/A"
    print(f"  FCD:           {fcd_str}")
    
    return samples_path, metrics
