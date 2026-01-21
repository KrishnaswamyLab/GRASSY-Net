"""
GRASSY Training Script with Learnable Scattering Transform (End-to-End)

This script trains the GRASSY model with a learnable graph scattering transform.
The scattering transform now contains learnable MLPs instead of fixed absolute values,
allowing the entire pipeline to be trained end-to-end.

Usage:
    python train_grassy_end_to_end.py                                    # Use default config
    python train_grassy_end_to_end.py --config my_config.yaml            # Use custom config
    python train_grassy_end_to_end.py --config config.yaml --override training.n_epochs=50
"""

import os
import datetime
import argparse
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.loader import DataLoader


from models.GRASSY_model import GRASSY
from models.ScatteringTransform import GraphScatteringTransform
from models.EndToEndWrapper import EndToEndScatteringGRASSYWrapper
from datasets.load_ZINC_tranche import ZINCDataset

from utils.config_utils import config_to_hparams, load_config, apply_overrides, get_grassy_flags
import yaml


def main():
    parser = argparse.ArgumentParser(description='Train GRASSY with learnable scattering (end-to-end)')
    parser.add_argument('--config', type=str, default='grassy_config.yaml',
                        help='Path to config file (default: grassy_config.yaml)')
    parser.add_argument('--override', type=str, nargs='*', default=[],
                        help='Override config values (e.g., training.n_epochs=50)')
    args = parser.parse_args()

    # Load and process config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    if args.override:
        config = apply_overrides(config, args.override)

    # Extract config sections
    dataset_cfg = config['dataset']
    model_cfg = config['model']
    training_cfg = config['training']
    hardware_cfg = config['hardware']
    logging_cfg = config['logging']
    scattering_cfg = config['scattering']
    early_stopping_cfg = config.get('early_stopping', {'enabled': False})

    # Get GRASSY version flags
    grassy_version = training_cfg['grassy_version']
    kl_div, reg = get_grassy_flags(grassy_version)

    # Adjust alpha and beta based on GRASSY version
    alpha = training_cfg['alpha'] if reg else 0
    beta = training_cfg['beta'] if kl_div else 0

    print(f"\n{'='*60}")
    print(f"GRASSY Training with Learnable Scattering (End-to-End)")
    print(f"{'='*60}")
    print(f"\nGRASSY Version: {grassy_version}")
    print(f"  - KL Divergence: {'enabled' if kl_div else 'disabled'}")
    print(f"  - Regression: {'enabled' if reg else 'disabled'}")
    print(f"  - Alpha (reg weight): {alpha}")
    print(f"  - Beta (KL weight): {beta}")

    # Load dataset
    print(f"\nLoading dataset: {dataset_cfg['name']}")
    base_dataset = ZINCDataset(
        dataset_cfg['path'],
        prop_stat_dict=dataset_cfg.get('stats_path'),
        transform=None,
        include_ki=dataset_cfg.get('include_ki', False)
    )
    print(f"Loaded {len(base_dataset)} molecules")
    print(f"Node features: {base_dataset.num_node_features}")
    print(f"Properties: {base_dataset.num_classes}")

    # Create learnable scattering transform
    print(f"\nScattering configuration (LEARNABLE):")
    print(f"  - Wavelet scales (J): {scattering_cfg['J']}")
    print(f"  - Moments: {scattering_cfg['num_moments']}")
    print(f"  - MLP hidden dim: {scattering_cfg.get('mlp_hidden_dim', 64)}")

    scattering_transform = GraphScatteringTransform(
        in_channels=base_dataset.num_node_features,
        J=scattering_cfg['J'],
        num_moments=scattering_cfg['num_moments'],
        mlp_hidden_dim=scattering_cfg.get('mlp_hidden_dim', 64),
    )
    scattering_dim = scattering_transform.out_shape()
    print(f"  - Output dimension: {scattering_dim}")

    # Data splits
    train_size = dataset_cfg['train_size']
    val_size = dataset_cfg['val_size']
    test_size = len(base_dataset) - train_size - val_size

    print(f"\nDataset splits: train={train_size}, val={val_size}, test={test_size}")

    train_set, val_set, test_set = torch.utils.data.random_split(
        base_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(dataset_cfg['seed'])
    )

    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        num_workers=training_cfg['num_workers'],
        drop_last=True
    )
    valid_loader = DataLoader(
        val_set,
        batch_size=training_cfg['batch_size'],
        shuffle=False,
        num_workers=training_cfg['num_workers']
    )
    test_loader = DataLoader(
        test_set,
        batch_size=training_cfg['batch_size'],
        shuffle=False,
        num_workers=training_cfg['num_workers'],
    )

    # Setup logging directory
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%H-%M-%S")
    reg_str = 'regress' if reg else 'noregress'
    kl_str = 'kld' if kl_div else 'nokld'
    save_dir = os.path.join(
        logging_cfg['save_dir'],
        f"{dataset_cfg['name']}_e2e_{reg_str}_{kl_str}_{date_suffix}/"
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"\nSave directory: {save_dir}")

    # Save config to output directory for reproducibility
    config_save_path = os.path.join(save_dir, 'config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to: {config_save_path}")

    # Setup logger
    logger = None
    if logging_cfg['wandb']['enabled']:
        wandb_cfg = logging_cfg['wandb']
        logger = WandbLogger(
            project=wandb_cfg['project'],
            entity=wandb_cfg['entity'],
            name=f"{dataset_cfg['name']}_e2e_{reg_str}_{kl_str}",
            save_dir=save_dir,
        )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_cfg = logging_cfg['checkpoint']
    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename='best-{epoch}-{val_loss:.3f}',
        monitor=checkpoint_cfg['monitor'],
        mode=checkpoint_cfg['mode'],
        save_top_k=checkpoint_cfg['save_top_k'],
        save_last=checkpoint_cfg['save_last'],
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback (optional)
    if early_stopping_cfg.get('enabled', False):
        early_stop_callback = EarlyStopping(
            monitor=early_stopping_cfg['monitor'],
            min_delta=early_stopping_cfg['min_delta'],
            patience=early_stopping_cfg['patience'],
            verbose=True,
            mode=early_stopping_cfg['mode']
        )
        callbacks.append(early_stop_callback)
        print("Early stopping: enabled")
    else:
        print("Early stopping: disabled")

    # Get input dimensions from first sample
    num_properties = base_dataset.num_classes
    len_epoch = len(train_loader)

    print(f"\nModel dimensions:")
    print(f"  - Scattering output dim: {scattering_dim}")
    print(f"  - Bottleneck dim: {model_cfg['bottle_dim']}")
    print(f"  - Hidden dim: {model_cfg['hidden_dim']}")
    print(f"  - Num properties: {num_properties}")
    print(f"  - Steps per epoch: {len_epoch}")

    # Create hparams and GRASSY model
    hparams = config_to_hparams(config, scattering_dim, num_properties, len_epoch)
    hparams.alpha = alpha
    hparams.beta = beta

    grassy_model = GRASSY(hparams=hparams)

    # Create end-to-end wrapper
    model = EndToEndScatteringGRASSYWrapper(scattering_transform, grassy_model, hparams, alpha, beta)

    # Log hyperparameters
    if logger:
        logger.log_hyperparams({
            'config': config,
            'scattering_dim': scattering_dim,
            'num_properties': num_properties,
            'grassy_version': grassy_version,
            'scattering_J': scattering_cfg['J'],
            'scattering_moments': scattering_cfg['num_moments'],
            'mlp_hidden_dim': scattering_cfg.get('mlp_hidden_dim', 64),
        })

    # Setup trainer
    trainer = pl.Trainer(
        max_epochs=training_cfg['n_epochs'],
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        accelerator=hardware_cfg.get('accelerator', 'auto'),
    )

    # Train
    print("\nStarting end-to-end training...")
    resume_path = config.get('resume_from_checkpoint')
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=resume_path,
    )

    # Save losses
    print("\nSaving loss histories...")
    loss = np.array(model.get_loss_list())
    prefix = f"{dataset_cfg['name']}_e2e_{reg_str}_{kl_str}"

    np.save(os.path.join(save_dir, f"{prefix}_total_loss_list.npy"), loss)
    np.save(os.path.join(save_dir, f"{prefix}_recon_loss_list.npy"), np.array(model.get_recon_loss_list()))
    np.save(os.path.join(save_dir, f"{prefix}_reg_loss_list.npy"), np.array(model.get_reg_loss_list()))
    np.save(os.path.join(save_dir, f"{prefix}_kl_loss_list.npy"), np.array(model.get_kl_loss_list()))

    # Save best model
    print("\nSaving model...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        checkpoint = torch.load(best_model_path)
        torch.save(checkpoint['state_dict'], os.path.join(save_dir, f"{prefix}_model.pt"))
        print(f"Saved best model from: {best_model_path}")
    else:
        torch.save(model.state_dict(), os.path.join(save_dir, f"{prefix}_model.pt"))
        print("Saved final model")

    # Save model to wandb
    if logger:
        logger.experiment.save(os.path.join(save_dir, f"{prefix}_model.pt"))

    print("Generating embeddings for full dataset...")

    scat_mom_list = []
    prop = [[] for _ in range(base_dataset.num_classes)]

    # Atom percentages
    carbon = []
    nitro = []
    oxy = []
    atom_percentage = [carbon, nitro, oxy]

    for index in tqdm(range(len(full_dataset))):
        entry = full_dataset[index]
        scat_mom_list.append(entry[0].detach().cpu().numpy())
        
        # Save all properties
        for i in range(len(entry[1])):
            val = entry[1][i].item() if torch.is_tensor(entry[1][i]) else entry[1][i]
            prop[i].append(val)
        
        # Get atom counts from original dataset
        data = no_transform_dataset[index]
        
        c, n, o = 0, 0, 0
        atom_count = 0
        
        for atom in data.element:
            if atom == 'C':
                c += 1
            elif atom == 'N':
                n += 1
            elif atom == 'O':
                o += 1
            atom_count += 1
        
        carbon.append(c / atom_count if atom_count > 0 else 0)
        nitro.append(n / atom_count if atom_count > 0 else 0)
        oxy.append(o / atom_count if atom_count > 0 else 0)

    scat_mom_list = np.array(scat_mom_list)
    print(f"Scattering coefficients shape: {scat_mom_list.shape}")

    print(f"\nTraining complete! Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
