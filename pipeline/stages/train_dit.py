"""
Stage 5: DiT (Diffusion Transformer) Training

Trains the GRASSY-DiT model for molecular generation.
Wraps grassy_dit/train.py functionality.
"""

import os
import sys
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import yaml
import numpy as np
import pandas as pd
import torch
import wandb
from rdkit import Chem

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from grassy_dit.train import ScatteringGraphDIT


def run_train_dit(
    train_dir: str,
    val_dir: Optional[str],
    output_dir: str,
    checkpoint: Optional[str] = None,
    epochs: int = 2000,
    hidden_size: int = 1024,
    num_layer: int = 6,
    num_head: int = 16,
    batch_size: int = 64,
    learning_rate: float = 2e-4,
    noise_prob: float = 0.2,
    noise_std: float = 0.2,
    noise_lower: float = 0.25,
    noise_upper: float = 1.0,
    J: int = 4,
    num_moments: int = 4,
    device: str = "auto",
    wandb_enabled: bool = False,
    wandb_project: str = "GRASSY-Pipeline",
    wandb_entity: str = "grassy",
    use_moment_tokens: bool = False,
    # Multi-phase training (optional)
    phase_epochs: Optional[str] = None,  # e.g. "1000,500,500"
    phase_lrs: Optional[str] = None,     # e.g. "2e-4,1e-4,5e-5"
    stage3_base_lr_ratio: Optional[float] = None,  # In Stage 3, base model gets lr*ratio
) -> Tuple[str, Dict[str, Any]]:
    """
    Run DiT training stage.
    
    Args:
        train_dir: Directory containing training data (molecules.csv, scattering_moments.npy)
        val_dir: Directory containing validation data (optional)
        output_dir: Directory to save outputs
        checkpoint: Path to existing checkpoint (for resuming)
        epochs: Number of epochs to train (0 = skip if checkpoint provided)
        hidden_size: Transformer hidden dimension
        num_layer: Number of transformer layers
        num_head: Number of attention heads
        batch_size: Training batch size
        learning_rate: Learning rate
        noise_prob: Probability of adding noise during training
        noise_std: Gaussian noise standard deviation
        noise_lower: Min fraction of moments to corrupt
        noise_upper: Max fraction of moments to corrupt
        J: Number of wavelet scales
        num_moments: Number of statistical moments
        device: Device to use
        wandb_enabled: Whether to enable W&B logging
        wandb_project: W&B project name
        wandb_entity: W&B entity name
    
    Returns:
        Tuple of (checkpoint_path, metrics_dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Stage 5: DiT (Diffusion Transformer) Training")
    print("=" * 60)
    
    # Check if we should skip training
    if epochs == 0:
        if checkpoint is None:
            raise ValueError("Cannot skip DiT training (epochs=0) without checkpoint")
        print(f"\nSkipping training (epochs=0), using provided checkpoint:")
        print(f"  {checkpoint}")
        return checkpoint, {'status': 'skipped', 'checkpoint_source': 'provided'}
    
    # Load training data
    print(f"\nLoading training data from: {train_dir}")
    df = pd.read_csv(os.path.join(train_dir, 'molecules.csv'))
    smiles = df['smiles'].tolist()
    scattering = np.load(os.path.join(train_dir, 'scattering_moments.npy'))
    
    # Filter incompatible molecules (dative bonds not supported)
    valid_smiles = []
    valid_scatter = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        has_dative = any(b.GetBondType() == Chem.BondType.DATIVE for b in mol.GetBonds())
        if not has_dative:
            valid_smiles.append(smi)
            valid_scatter.append(scattering[i])
    
    smiles = valid_smiles
    scattering = np.array(valid_scatter)
    print(f"Training: {len(smiles)} molecules after filtering")
    
    # Load validation data if provided
    val_smiles, val_scattering = None, None
    if val_dir and os.path.exists(os.path.join(val_dir, 'molecules.csv')):
        print(f"Loading validation data from: {val_dir}")
        df_val = pd.read_csv(os.path.join(val_dir, 'molecules.csv'))
        val_smiles_raw = df_val['smiles'].tolist()
        val_scattering_raw = np.load(os.path.join(val_dir, 'scattering_moments.npy'))
        
        # Filter validation molecules
        valid_val_smiles, valid_val_scatter = [], []
        for i, smi in enumerate(val_smiles_raw):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            has_dative = any(b.GetBondType() == Chem.BondType.DATIVE for b in mol.GetBonds())
            if not has_dative:
                valid_val_smiles.append(smi)
                valid_val_scatter.append(val_scattering_raw[i])
        
        val_smiles = valid_val_smiles
        val_scattering = np.array(valid_val_scatter)
        print(f"Validation: {len(val_smiles)} molecules after filtering")
    
    # Build config for DiT
    config = {
        'dataset': {
            'name': 'pipeline',
            'data_dir': train_dir,
            'csv_file': 'molecules.csv',
            'smiles_col': 'smiles',
            'scatter_file': 'scattering_moments.npy',
            'seed': 42,
            'val_data_dir': val_dir,
            'val_every_n_epochs': 5,
        },
        'scattering': {
            'J': J,
            'num_moments': num_moments,
        },
        'augmentation': {
            'moment_noise': {
                'prob': noise_prob,
                'lower_scalar': noise_lower,
                'upper_scalar': noise_upper,
                'noise_std': noise_std,
            }
        },
        'model': {
            'max_node': 50,  # Will be determined from data
            'hidden_size': hidden_size,
            'num_layer': num_layer,
            'num_head': num_head,
            'mlp_ratio': 4.0,
            'use_moment_tokens': use_moment_tokens,
        },
        'training': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'phase_epochs': phase_epochs,  # Multi-phase: e.g. "1000,500,500"
            'phase_lrs': phase_lrs,        # Multi-phase: e.g. "2e-4,1e-4,5e-5"
            'stage3_base_lr_ratio': stage3_base_lr_ratio,  # In Stage 3, base params get lr*ratio
        },
        'checkpoint': {
            'save_dir': output_dir,
            'save_best': True,
            'save_every_n_epochs': 10,
            'resume_from': checkpoint,
        },
        'hardware': {
            'device': device,
            'num_workers': 4,
            'pin_memory': True,
        },
        'logging': {
            'wandb': {
                'enabled': wandb_enabled,
                'project': wandb_project,
                'entity': wandb_entity,
            },
            'verbose': True,
        },
    }
    
    # Save config
    config_path = os.path.join(output_dir, 'dit_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"\nConfig saved to: {config_path}")
    
    # Initialize W&B (always init, use disabled mode if not enabled)
    # This prevents "wandb.init() not called" errors in the underlying code
    wandb.init(
        mode="online" if wandb_enabled else "disabled",
        project=wandb_project,
        entity=wandb_entity,
        name=f"DiT_h{hidden_size}_l{num_layer}_e{epochs}",
        config=config,
    )
    
    # Create model
    cross_attn_drop = config.get('model', {}).get('cross_attn_drop', 0.0)
    print(f"\nInitializing model:")
    print(f"  - Hidden size: {hidden_size}")
    print(f"  - Layers: {num_layer}")
    print(f"  - Heads: {num_head}")
    print(f"  - Noise prob: {noise_prob}")
    print(f"  - Noise std: {noise_std}")
    print(f"  - Cross-attn dropout: {cross_attn_drop}")
    print(f"  - Moment tokens: {use_moment_tokens}")
    
    model = ScatteringGraphDIT(config=config)
    
    # Compute num_atom_types from scattering
    num_levels = 1 + J + J * (J - 1) // 2
    scattering_dim = scattering.shape[-1]
    num_atom_types = scattering_dim // (num_levels * num_moments)
    model.num_atom_types = num_atom_types
    model.num_levels = num_levels
    model.num_moments = num_moments
    
    # Load checkpoint if resuming
    if checkpoint and os.path.exists(checkpoint):
        print(f"\nResuming from checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location='cpu')
        model._initialize_model(None, checkpoint=ckpt)
    
    # Train
    print(f"\nStarting training for {epochs} epochs...")
    model.fit(
        X_train=smiles,
        y_train=scattering,
        X_val=val_smiles,
        y_val=val_scattering,
        val_every_n_epochs=5,
    )
    
    # Get best checkpoint path
    best_checkpoint = os.path.join(output_dir, 'checkpoint_best.pt')
    if not os.path.exists(best_checkpoint):
        # Save final model
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        best_checkpoint = os.path.join(output_dir, f'final_model_{timestamp}.pt')
        model.save_to_local(best_checkpoint)
    
    print(f"\nSaved checkpoint: {best_checkpoint}")
    
    # Cleanup W&B
    if wandb_enabled and wandb.run is not None:
        wandb.finish()
    
    # Metrics
    metrics = {
        'status': 'trained',
        'checkpoint_source': 'trained',
        'epochs_trained': epochs,
        'hidden_size': hidden_size,
        'num_layer': num_layer,
        'num_head': num_head,
        'train_molecules': len(smiles),
        'val_molecules': len(val_smiles) if val_smiles else 0,
        'scattering_dim': scattering_dim,
        'num_atom_types': num_atom_types,
    }
    
    print(f"\nDiT training complete!")
    return best_checkpoint, metrics
