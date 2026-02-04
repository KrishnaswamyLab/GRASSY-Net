"""
Train the moment classifier on noisy molecular graphs.

The classifier learns to predict scattering moments from noisy graphs at
various timesteps. Once trained, it can be used for classifier guidance
during molecule generation.

Usage:
    # Basic training
    python -m guided_diffusion.train_classifier \
        --data_path guided_diffusion/classifier_data/classifier_training_data.pt \
        --output_dir guided_diffusion/checkpoints/moment_classifier

    # With DiT weight initialization
    python -m guided_diffusion.train_classifier \
        --data_path guided_diffusion/classifier_data/classifier_training_data.pt \
        --output_dir guided_diffusion/checkpoints/moment_classifier \
        --init_from_dit path/to/dit_checkpoint.pt

    # Freeze encoder (train only output head)
    python -m guided_diffusion.train_classifier \
        --data_path guided_diffusion/classifier_data/classifier_training_data.pt \
        --init_from_dit path/to/dit_checkpoint.pt \
        --freeze_encoder
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from guided_diffusion.prepare_classifier_data import ClassifierDataset
from guided_diffusion.moment_classifier import (
    MomentClassifier, 
    load_classifier, 
    init_from_dit_checkpoint
)


def compute_timestep_buckets(timesteps: torch.Tensor, num_timesteps: int, num_buckets: int = 3):
    """
    Assign timesteps to buckets for per-bucket loss tracking.
    
    Args:
        timesteps: [B] timestep values
        num_timesteps: Total number of timesteps (T)
        num_buckets: Number of buckets (default 3: early/mid/late)
    
    Returns:
        [B] bucket indices (0 to num_buckets-1)
    """
    # Normalize timesteps to [0, 1]
    t_norm = timesteps.float() / num_timesteps
    
    # Assign to buckets
    bucket_size = 1.0 / num_buckets
    buckets = (t_norm / bucket_size).long().clamp(0, num_buckets - 1)
    
    return buckets


def train_epoch(
    model: MomentClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_timesteps: int,
    epoch: int,
    log_wandb: bool = True,
):
    """
    Train for one epoch.
    
    Returns:
        dict with loss statistics
    """
    model.train()
    
    total_loss = 0.0
    num_batches = 0
    
    # Track loss per timestep bucket (early/mid/late)
    bucket_losses = {0: [], 1: [], 2: []}
    bucket_names = ['early', 'mid', 'late']
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (noisy_X, noisy_E, t, node_mask, clean_moments) in enumerate(pbar):
        # Move to device
        noisy_X = noisy_X.to(device).float()
        noisy_E = noisy_E.to(device).float()
        t = t.to(device)
        node_mask = node_mask.to(device)
        clean_moments = clean_moments.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_moments = model(noisy_X, noisy_E, t, node_mask)
        
        # MSE loss
        loss = F.mse_loss(pred_moments, clean_moments)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        # Track per-bucket loss (disabled - tensor indexing issue)
        # buckets = compute_timestep_buckets(t, num_timesteps)
        # with torch.no_grad():
        #     per_sample_loss = ((pred_moments - clean_moments) ** 2).mean(dim=-1)
        #     for b in range(3):
        #         mask = buckets == b
        #         if mask.any():
        #             bucket_losses[b].append(per_sample_loss[mask].mean().item())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
        
        # Log to wandb
        if log_wandb and batch_idx % 100 == 0:
            step = epoch * len(dataloader) + batch_idx
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/step': step,
            })
    
    avg_loss = total_loss / num_batches
    
    # Compute average bucket losses
    bucket_avg = {}
    for b, name in enumerate(bucket_names):
        if bucket_losses[b]:
            bucket_avg[f'loss_{name}'] = np.mean(bucket_losses[b])
        else:
            bucket_avg[f'loss_{name}'] = 0.0
    
    return {
        'loss': avg_loss,
        **bucket_avg,
    }


@torch.no_grad()
def validate(
    model: MomentClassifier,
    dataloader: DataLoader,
    device: torch.device,
    num_timesteps: int,
):
    """
    Validate the model.
    
    Returns:
        dict with validation statistics
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    bucket_losses = {0: [], 1: [], 2: []}
    bucket_names = ['early', 'mid', 'late']
    
    for noisy_X, noisy_E, t, node_mask, clean_moments in dataloader:
        noisy_X = noisy_X.to(device).float()
        noisy_E = noisy_E.to(device).float()
        t = t.to(device)
        node_mask = node_mask.to(device)
        clean_moments = clean_moments.to(device)
        
        pred_moments = model(noisy_X, noisy_E, t, node_mask)
        loss = F.mse_loss(pred_moments, clean_moments)
        
        total_loss += loss.item()
        num_batches += 1
        
        # Per-bucket tracking (disabled - tensor indexing issue)
        # buckets = compute_timestep_buckets(t, num_timesteps)
        # per_sample_loss = ((pred_moments - clean_moments) ** 2).mean(dim=-1)
        # for b in range(3):
        #     mask = buckets == b
        #     if mask.any():
        #         bucket_losses[b].append(per_sample_loss[mask].mean().item())
    
    avg_loss = total_loss / num_batches
    
    bucket_avg = {}
    for b, name in enumerate(bucket_names):
        if bucket_losses[b]:
            bucket_avg[f'loss_{name}'] = np.mean(bucket_losses[b])
        else:
            bucket_avg[f'loss_{name}'] = 0.0
    
    return {
        'loss': avg_loss,
        **bucket_avg,
    }


def train_classifier(
    data_path: str,
    output_dir: str,
    hidden_size: int = 256,
    num_layers: int = 4,
    num_heads: int = 8,
    mlp_ratio: float = 4.0,
    dropout: float = 0.1,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    epochs: int = 100,
    val_split: float = 0.1,
    init_from_dit: str = None,
    freeze_encoder: bool = False,
    save_every: int = 10,
    wandb_project: str = 'GRASSY-Classifier',
    wandb_enabled: bool = True,
    device: str = 'auto',
):
    """
    Train the moment classifier.
    
    Args:
        data_path: Path to classifier_training_data.pt
        output_dir: Directory to save checkpoints
        hidden_size: Transformer hidden dimension
        num_layers: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden size multiplier
        dropout: Dropout rate
        batch_size: Training batch size
        learning_rate: Learning rate
        epochs: Number of epochs
        val_split: Fraction of data for validation
        init_from_dit: Path to DiT checkpoint for weight initialization
        freeze_encoder: Freeze encoder weights (train only head)
        save_every: Save checkpoint every N epochs
        wandb_project: Weights & Biases project name
        wandb_enabled: Whether to log to wandb
        device: Device to use ('auto', 'cuda', 'cpu')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Device setup
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading data from {data_path}...")
    dataset = ClassifierDataset(data_path)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Metadata: {dataset.metadata}")
    
    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train size: {train_size}, Val size: {val_size}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    # Initialize model
    print("Initializing model...")
    model = MomentClassifier(
        max_n_nodes=dataset.max_node,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        Xdim=dataset.Xdim,
        Edim=dataset.Edim,
        moment_dim=dataset.moment_dim,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Optional: Initialize from DiT checkpoint
    if init_from_dit:
        print(f"Initializing from DiT checkpoint: {init_from_dit}")
        model = init_from_dit_checkpoint(model, init_from_dit, device, freeze_encoder)
        
        # Recount trainable params after freezing
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable parameters after init: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01,
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate * 0.01
    )
    
    # Initialize wandb
    if wandb_enabled:
        run_name = f"classifier_h{hidden_size}_L{num_layers}_e{epochs}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'num_heads': num_heads,
                'mlp_ratio': mlp_ratio,
                'dropout': dropout,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'val_split': val_split,
                'init_from_dit': init_from_dit,
                'freeze_encoder': freeze_encoder,
                'max_node': dataset.max_node,
                'Xdim': dataset.Xdim,
                'Edim': dataset.Edim,
                'moment_dim': dataset.moment_dim,
                'train_size': train_size,
                'val_size': val_size,
            }
        )
    
    # Training loop
    best_val_loss = float('inf')
    num_timesteps = dataset.num_timesteps
    
    print(f"\nStarting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        # Train
        train_stats = train_epoch(
            model, train_loader, optimizer, device, num_timesteps, epoch, wandb_enabled
        )
        
        # Validate
        val_stats = validate(model, val_loader, device, num_timesteps)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Print stats
        print(f"Epoch {epoch}/{epochs}")
        print(f"  Train Loss: {train_stats['loss']:.6f} "
              f"(early: {train_stats['loss_early']:.4f}, "
              f"mid: {train_stats['loss_mid']:.4f}, "
              f"late: {train_stats['loss_late']:.4f})")
        print(f"  Val Loss:   {val_stats['loss']:.6f} "
              f"(early: {val_stats['loss_early']:.4f}, "
              f"mid: {val_stats['loss_mid']:.4f}, "
              f"late: {val_stats['loss_late']:.4f})")
        print(f"  LR: {current_lr:.2e}")
        
        # Log to wandb
        if wandb_enabled:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_stats['loss'],
                'train/loss_early': train_stats['loss_early'],
                'train/loss_mid': train_stats['loss_mid'],
                'train/loss_late': train_stats['loss_late'],
                'val/loss': val_stats['loss'],
                'val/loss_early': val_stats['loss_early'],
                'val/loss_mid': val_stats['loss_mid'],
                'val/loss_late': val_stats['loss_late'],
                'lr': current_lr,
            })
        
        # Save best checkpoint
        if val_stats['loss'] < best_val_loss:
            best_val_loss = val_stats['loss']
            save_checkpoint(
                model, optimizer, epoch, val_stats['loss'],
                dataset, hidden_size, num_layers, num_heads, mlp_ratio, dropout,
                os.path.join(output_dir, 'checkpoint_best.pt')
            )
            print(f"  New best! Saved checkpoint.")
        
        # Save periodic checkpoint
        if epoch % save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, val_stats['loss'],
                dataset, hidden_size, num_layers, num_heads, mlp_ratio, dropout,
                os.path.join(output_dir, f'checkpoint_epoch{epoch}.pt')
            )
    
    # Save final checkpoint
    save_checkpoint(
        model, optimizer, epochs, val_stats['loss'],
        dataset, hidden_size, num_layers, num_heads, mlp_ratio, dropout,
        os.path.join(output_dir, 'checkpoint_final.pt')
    )
    
    print(f"\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {output_dir}")
    
    if wandb_enabled:
        wandb.finish()
    
    return model


def save_checkpoint(
    model: MomentClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    dataset: ClassifierDataset,
    hidden_size: int,
    num_layers: int,
    num_heads: int,
    mlp_ratio: float,
    dropout: float,
    path: str,
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': {
            'max_n_nodes': dataset.max_node,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'Xdim': dataset.Xdim,
            'Edim': dataset.Edim,
            'moment_dim': dataset.moment_dim,
            'mlp_ratio': mlp_ratio,
            'dropout': dropout,
        },
        'metadata': dataset.metadata,
    }
    torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(
        description='Train moment classifier for guided diffusion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to classifier_training_data.pt')
    parser.add_argument('--output_dir', type=str, 
                        default='guided_diffusion/checkpoints/moment_classifier',
                        help='Output directory for checkpoints')
    
    # Model arguments
    parser.add_argument('--hidden_size', type=int, default=256,
                        help='Transformer hidden dimension')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='Number of transformer blocks')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0,
                        help='MLP hidden size multiplier')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of data for validation')
    
    # Transfer learning arguments
    parser.add_argument('--init_from_dit', type=str, default=None,
                        help='Path to DiT checkpoint for weight initialization')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder weights (train only output head)')
    
    # Logging arguments
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--wandb_project', type=str, default='GRASSY-Classifier',
                        help='Weights & Biases project name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable wandb logging')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        help="Device to use ('auto', 'cuda', 'cpu')")
    
    args = parser.parse_args()
    
    train_classifier(
        data_path=args.data_path,
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        val_split=args.val_split,
        init_from_dit=args.init_from_dit,
        freeze_encoder=args.freeze_encoder,
        save_every=args.save_every,
        wandb_project=args.wandb_project,
        wandb_enabled=not args.no_wandb,
        device=args.device,
    )


if __name__ == '__main__':
    main()
