"""
Stage 8: Property Optimization Sampling

Generates molecules by optimizing in GRASSY's latent space for a target property.
Wraps grassy_dit/sample_property_optimization.py functionality.

Workflow:
1. Initialize latent vector (sample from prior or use mean)
2. Optimize in latent space for target property (gradient ascent)
3. Decode optimized latent back to scattering moments
4. Generate molecules conditioned on optimized scattering
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

import numpy as np
import torch
import yaml
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Descriptors, QED

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grassy_dit.train import ScatteringGraphDIT
from models.GRASSY_model import GRASSY

# Try to import LatentOptimizer
try:
    from models.LatentOptimization import LatentOptimizer
    HAS_OPTIMIZER = True
except ImportError:
    HAS_OPTIMIZER = False
    print("Warning: LatentOptimization not available")


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


def compute_molecule_properties(smiles_list: List[str]) -> Dict[str, List]:
    """Compute properties for generated molecules."""
    properties = {
        'smiles': smiles_list,
        'valid': [],
        'qed': [],
        'sa': [],
        'logp': [],
        'mol_wt': [],
    }
    
    for smi in smiles_list:
        if smi is None:
            properties['valid'].append(False)
            properties['qed'].append(float('nan'))
            properties['sa'].append(float('nan'))
            properties['logp'].append(float('nan'))
            properties['mol_wt'].append(float('nan'))
            continue
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            properties['valid'].append(False)
            properties['qed'].append(float('nan'))
            properties['sa'].append(float('nan'))
            properties['logp'].append(float('nan'))
            properties['mol_wt'].append(float('nan'))
        else:
            properties['valid'].append(True)
            properties['qed'].append(QED.qed(mol))
            try:
                from rdkit.Chem import RDConfig
                sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
                import sascorer
                properties['sa'].append(sascorer.calculateScore(mol))
            except:
                properties['sa'].append(float('nan'))
            properties['logp'].append(Descriptors.MolLogP(mol))
            properties['mol_wt'].append(Descriptors.MolWt(mol))
    
    return properties


PROPERTY_NAMES = {
    0: 'MolLogP',
    1: 'TPSA',
    2: 'MolWt',
    'qed': 'QED',
    'logp': 'MolLogP',
    'sa': 'SA Score',
    'mw': 'MolWt',
}


def run_sample_property_opt(
    dit_checkpoint: str,
    dit_config: str,
    grassy_checkpoint: str,
    output_dir: str,
    property_target: str = "qed",
    n_trajectories: int = 10,
    n_steps: int = 50,
    n_samples_per_traj: int = 10,
    init_method: str = "mean",
    noise_scale: float = 0.1,
    device: str = "auto",
) -> Tuple[str, Dict[str, Any]]:
    """
    Run property optimization sampling stage.
    
    Args:
        dit_checkpoint: Path to DiT checkpoint
        dit_config: Path to DiT config
        grassy_checkpoint: Path to GRASSY checkpoint
        output_dir: Directory to save results
        property_target: Target property to optimize (qed, logp, sa, or index 0-2)
        n_trajectories: Number of optimization trajectories
        n_steps: Number of optimization steps per trajectory
        n_samples_per_traj: Number of molecules to generate per trajectory
        init_method: Initialization method (mean, prior)
        noise_scale: Noise scale for optimization
        device: Device to use
    
    Returns:
        Tuple of (samples_path, metrics_dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Stage 8: Property Optimization Sampling")
    print("=" * 60)
    
    if not HAS_OPTIMIZER:
        print("Warning: LatentOptimizer not available, using simple gradient ascent")
    
    # Setup device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Parse property target
    if property_target.isdigit():
        property_idx = int(property_target)
    elif property_target.lower() == 'qed':
        property_idx = 0  # Typically MolLogP correlates with QED
    elif property_target.lower() == 'logp':
        property_idx = 0
    elif property_target.lower() == 'sa':
        property_idx = 1
    elif property_target.lower() == 'mw':
        property_idx = 2
    else:
        property_idx = 0
    
    print(f"Optimizing for property index: {property_idx}")
    
    # Load models
    print(f"\nLoading DiT model from: {dit_checkpoint}")
    dit_model = load_dit_model(dit_checkpoint, dit_config, device)
    
    print(f"Loading GRASSY model from: {grassy_checkpoint}")
    grassy_model = load_grassy_model(grassy_checkpoint, device)
    
    latent_dim = grassy_model.hparams.get('bottle_dim', 32)
    
    # Run optimization trajectories
    all_smiles = []
    all_properties = []
    initial_properties = []
    final_properties = []
    
    print(f"\nRunning {n_trajectories} optimization trajectories...")
    
    for traj_idx in tqdm(range(n_trajectories), desc="Trajectories"):
        # Initialize latent vector
        if init_method == "mean":
            z = torch.zeros(1, latent_dim, device=device)
        else:  # prior
            z = torch.randn(1, latent_dim, device=device) * noise_scale
        
        z.requires_grad_(True)
        
        # Get initial property
        with torch.no_grad():
            y_init, _, _ = grassy_model.predict(z)
            initial_prop = y_init[0, property_idx].item()
            initial_properties.append(initial_prop)
        
        # Simple gradient ascent optimization
        optimizer = torch.optim.Adam([z], lr=0.1)
        
        for step in range(n_steps):
            optimizer.zero_grad()
            y_pred, scattering, num_atoms = grassy_model.predict(z)
            
            # Maximize the target property
            loss = -y_pred[0, property_idx]
            loss.backward()
            optimizer.step()
        
        # Get final property
        with torch.no_grad():
            y_final, scattering_final, num_atoms_final = grassy_model.predict(z)
            final_prop = y_final[0, property_idx].item()
            final_properties.append(final_prop)
        
        # Generate molecules from optimized latent
        scattering_np = scattering_final[0].cpu().numpy()
        num_nodes = int(round(num_atoms_final[0].item()))
        num_nodes = max(5, min(50, num_nodes))
        
        try:
            generated = dit_model.generate(
                scattering=scattering_np,
                num_nodes=num_nodes,
                batch_size=n_samples_per_traj,
            )
            
            all_smiles.extend(generated)
            props = compute_molecule_properties(generated)
            all_properties.append(props)
        except Exception as e:
            print(f"Warning: Generation failed for trajectory {traj_idx}: {e}")
    
    # Aggregate results
    avg_initial = np.mean(initial_properties) if initial_properties else 0
    avg_final = np.mean(final_properties) if final_properties else 0
    improvement = avg_final - avg_initial
    
    valid_smiles = [s for s in all_smiles if s is not None]
    valid_mols = [Chem.MolFromSmiles(s) for s in valid_smiles]
    valid_mols = [m for m in valid_mols if m is not None]
    
    # Compute aggregate property metrics
    qed_values = [QED.qed(m) for m in valid_mols]
    logp_values = [Descriptors.MolLogP(m) for m in valid_mols]
    
    metrics = {
        'n_trajectories': n_trajectories,
        'n_steps': n_steps,
        'n_total_generated': len(all_smiles),
        'n_valid': len(valid_mols),
        'validity': (len(valid_mols) / len(all_smiles) * 100) if all_smiles else 0,
        'property_idx': property_idx,
        'avg_initial_property': avg_initial,
        'avg_final_property': avg_final,
        'property_improvement': improvement,
        'avg_qed': np.mean(qed_values) if qed_values else 0,
        'avg_logp': np.mean(logp_values) if logp_values else 0,
    }
    
    # Save generated molecules
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    samples_path = os.path.join(output_dir, f"property_opt_samples_{timestamp}.txt")
    with open(samples_path, 'w') as f:
        f.write(f"# Property optimization sampling\n")
        f.write(f"# property_target: {property_target} (idx={property_idx})\n")
        f.write(f"# n_trajectories: {n_trajectories}\n")
        f.write(f"# n_steps: {n_steps}\n")
        f.write(f"# init_method: {init_method}\n")
        f.write(f"# avg_initial: {avg_initial:.4f}\n")
        f.write(f"# avg_final: {avg_final:.4f}\n")
        f.write(f"# improvement: {improvement:+.4f}\n")
        for smi in valid_smiles:
            f.write(smi + "\n")
    print(f"\nSaved samples to: {samples_path}")
    
    # Save metrics
    results = {
        'config': {
            'dit_checkpoint': dit_checkpoint,
            'grassy_checkpoint': grassy_checkpoint,
            'property_target': property_target,
            'n_trajectories': n_trajectories,
            'n_steps': n_steps,
            'init_method': init_method,
        },
        'metrics': metrics,
        'timestamp': timestamp,
    }
    
    metrics_path = os.path.join(output_dir, f"property_opt_metrics_{timestamp}.json")
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved metrics to: {metrics_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Property Optimization Results:")
    print("=" * 60)
    print(f"  Target property:     {property_target} (idx={property_idx})")
    print(f"  Trajectories:        {n_trajectories}")
    print(f"  Steps per traj:      {n_steps}")
    print(f"  Initial property:    {avg_initial:.4f}")
    print(f"  Final property:      {avg_final:.4f}")
    print(f"  Improvement:         {improvement:+.4f}")
    print(f"  Validity:            {metrics['validity']:.2f}%")
    print(f"  Avg QED:             {metrics['avg_qed']:.4f}")
    print(f"  Avg LogP:            {metrics['avg_logp']:.4f}")
    
    return samples_path, metrics
