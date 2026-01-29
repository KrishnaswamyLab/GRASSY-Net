"""
Stage 4: GRASSY Model Training

Trains the GRASSY autoencoder on scattering moments.
Wraps train_grassy_fixed_scattering.py functionality.
"""

import os
import sys
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from types import SimpleNamespace

import yaml
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.GRASSY_model import GRASSY
from datasets.ZINCDataset import ZINCDataset


class PrecomputedScatteringDataset(torch.utils.data.Dataset):
    """Load precomputed scattering coefficients."""
    
    def __init__(self, scattering_path: str, base_dataset):
        self.coefficients = torch.from_numpy(np.load(scattering_path)).float()
        
        self.properties = []
        for i in range(len(base_dataset)):
            y = base_dataset[i].y
            y = y.squeeze(0).float()
            y[-1] = y[-1] - 7  # Adjust num_atoms property
            self.properties.append(y)
        
        assert len(self.coefficients) == len(self.properties), \
            f"Mismatch: {len(self.coefficients)} coefficients vs {len(self.properties)} molecules"
    
    def __len__(self):
        return len(self.coefficients)
    
    def __getitem__(self, idx):
        return self.coefficients[idx], self.properties[idx]


def run_train_grassy(
    data_path: str,
    stats_path: str,
    scattering_path: str,
    output_dir: str,
    checkpoint: Optional[str] = None,
    epochs: int = 100,
    bottle_dim: int = 32,
    hidden_dim: int = 128,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    alpha: float = 0.01,
    train_pct: float = 80,
    val_pct: float = 10,
    test_pct: float = 10,
    seed: int = 42,
    device: str = "auto",
    wandb_enabled: bool = False,
    wandb_project: str = "GRASSY-Pipeline",
    wandb_entity: str = "grassy",
) -> Tuple[str, Dict[str, Any]]:
    """
    Run GRASSY training stage.
    
    Args:
        data_path: Path to dataset .npy file
        stats_path: Path to statistics .npy file
        scattering_path: Path to precomputed scattering_moments.npy
        output_dir: Directory to save outputs
        checkpoint: Path to existing checkpoint (for resuming)
        epochs: Number of epochs to train (0 = skip if checkpoint provided)
        bottle_dim: Bottleneck dimension
        hidden_dim: Hidden layer dimension
        batch_size: Training batch size
        learning_rate: Learning rate
        alpha: Regression loss weight
        train_pct: Training percentage
        val_pct: Validation percentage
        test_pct: Test percentage
        seed: Random seed for data splits
        device: Device to use
        wandb_enabled: Whether to enable W&B logging
        wandb_project: W&B project name
        wandb_entity: W&B entity name
    
    Returns:
        Tuple of (checkpoint_path, metrics_dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Stage 4: GRASSY Model Training")
    print("=" * 60)
    
    # Check if we should skip training
    if epochs == 0:
        if checkpoint is None:
            raise ValueError("Cannot skip GRASSY training (epochs=0) without checkpoint")
        print(f"\nSkipping training (epochs=0), using provided checkpoint:")
        print(f"  {checkpoint}")
        return checkpoint, {'status': 'skipped', 'checkpoint_source': 'provided'}
    
    # Load base dataset
    print(f"\nLoading dataset: {data_path}")
    base_dataset = ZINCDataset(data_path, prop_stat_dict=stats_path, transform=None)
    print(f"Loaded {len(base_dataset)} molecules")
    print(f"Node features: {base_dataset.num_node_features}")
    print(f"Properties: {base_dataset.num_classes}")
    
    # Load precomputed scattering
    print(f"\nLoading precomputed scattering: {scattering_path}")
    full_dataset = PrecomputedScatteringDataset(scattering_path, base_dataset)
    
    # Data splits
    total_size = len(full_dataset)
    train_size = int(total_size * train_pct / 100)
    val_size = int(total_size * val_pct / 100)
    test_size = total_size - train_size - val_size
    
    print(f"\nDataset splits: train={train_size}, val={val_size}, test={test_size}")
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )
    valid_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    # Get dimensions
    input_dim = len(train_set[0][0])
    num_properties = len(train_set[0][1])
    len_epoch = len(train_loader)
    
    print(f"\nModel dimensions:")
    print(f"  - Input dim: {input_dim}")
    print(f"  - Bottleneck dim: {bottle_dim}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Num properties: {num_properties}")
    
    # Create hparams
    hparams = SimpleNamespace(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        bottle_dim=bottle_dim,
        num_properties=num_properties,
        learning_rate=learning_rate,
        alpha=alpha,
        beta=0.0,
        len_epoch=len_epoch,
        n_epochs=epochs,
        n_gpus=1 if device != "cpu" and torch.cuda.is_available() else 0,
    )
    
    # Create model
    model = GRASSY(hparams=hparams)
    
    # Resume from checkpoint if provided
    if checkpoint:
        print(f"\nResuming from checkpoint: {checkpoint}")
        ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if 'state_dict' in ckpt:
            model.load_state_dict(ckpt['state_dict'])
        else:
            model.load_state_dict(ckpt)
    
    # Setup logger
    logger = None
    if wandb_enabled:
        logger = WandbLogger(
            project=wandb_project,
            entity=wandb_entity,
            name=f"GRASSY_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            save_dir=output_dir,
        )
    
    # Setup callbacks
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='best-{epoch}-{val_loss:.3f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.0,
        patience=10,
        verbose=True,
        mode='min'
    )
    callbacks.append(early_stop_callback)
    
    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        accelerator=device if device != "auto" else "auto",
    )
    
    # Train
    print(f"\nStarting training for {epochs} epochs...")
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )
    
    # Save final model
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        final_checkpoint = best_model_path
    else:
        final_checkpoint = os.path.join(output_dir, "last.ckpt")
        trainer.save_checkpoint(final_checkpoint)
    
    print(f"\nSaved checkpoint: {final_checkpoint}")
    
    # Metrics
    metrics = {
        'status': 'trained',
        'checkpoint_source': 'trained',
        'epochs_trained': epochs,
        'input_dim': input_dim,
        'bottle_dim': bottle_dim,
        'hidden_dim': hidden_dim,
        'best_val_loss': float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else None,
    }
    
    print(f"\nGRASSY training complete!")
    return final_checkpoint, metrics
