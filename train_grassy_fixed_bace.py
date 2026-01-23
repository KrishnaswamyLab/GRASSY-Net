"""
GRASSY Training Script with Fixed Scattering Transform (YAML Configuration)

This script trains the GRASSY model using the fixed (non-learnable) GraphScatteringTransform.
Since the scattering has no learnable parameters, coefficients are computed once before training.

Usage:
    python train_grassy_fixed_bace.py                                    # Use default config
    python train_grassy_fixed_bace.py --config my_config.yaml            # Use custom config
    python train_grassy_fixed_bace.py --config config.yaml --override training.n_epochs=50
"""

import os
import datetime
import argparse

import yaml
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

from models.GRASSY_model import GRASSY
from models.ScatteringTransform import GraphScatteringTransform
from datasets.ZINCDataset import ZINCDataset

from utils.config_utils import load_config, apply_overrides, config_to_hparams, get_grassy_flags


class FixedScatteringTransform:
    """
    Transform that applies fixed GraphScatteringTransform to PyG Data objects.

    No learnable parameters - computes scattering coefficients deterministically.
    """

    def __init__(self, in_channels, J=4, num_moments=4):
        self.scattering = GraphScatteringTransform(
            in_channels=in_channels,
            J=J,
            num_moments=num_moments,
        )
        self.scattering.eval()

    def __call__(self, data):
        """Apply scattering transform to a single graph."""
        with torch.no_grad():
            data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
            coeffs = self.scattering(data)
        return coeffs.squeeze(0).detach(), data.y.squeeze(0)

    def out_shape(self):
        return self.scattering.out_shape()


class ScatteringDataset(torch.utils.data.Dataset):
    """
    Dataset that pre-computes scattering coefficients.

    More efficient than on-the-fly computation since transform is deterministic.
    """

    def __init__(self, base_dataset, scattering_transform, show_progress=True):
        self.coefficients = []
        self.properties = []

        iterator = tqdm(base_dataset, desc="Computing scattering") if show_progress else base_dataset

        for data in iterator:
            coeffs, props = scattering_transform(data)
            self.coefficients.append(coeffs)
            self.properties.append(props)

    def __len__(self):
        return len(self.coefficients)

    def __getitem__(self, idx):
        return self.coefficients[idx], self.properties[idx]


def main():
    parser = argparse.ArgumentParser(description='Train GRASSY with fixed scattering (YAML configuration)')
    parser.add_argument('--config', type=str, default='fixed_bace_config.yaml',
                        help='Path to config file (default: fixed_bace_config.yaml)')
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
    print(f"GRASSY Training with Fixed Scattering Transform")
    print(f"{'='*60}")
    print(f"\nGRASSY Version: {grassy_version}")
    print(f"  - KL Divergence: {'enabled' if kl_div else 'disabled'}")
    print(f"  - Regression: {'enabled' if reg else 'disabled'}")
    print(f"  - Alpha (reg weight): {alpha}")
    print(f"  - Beta (KL weight): {beta}")

    # Load train, val, and test datasets from separate files
    print(f"\nLoading datasets: {dataset_cfg['name']}")
    
    train_path = dataset_cfg['train_path']
    val_path = dataset_cfg['val_path']
    test_path = dataset_cfg['test_path']
    stats_path = dataset_cfg.get('stats_path')
    include_ki = dataset_cfg.get('include_ki', False)

    print(f"  - Train: {train_path}")
    print(f"  - Val: {val_path}")
    print(f"  - Test: {test_path}")

    train_base_dataset = ZINCDataset(
        train_path,
        prop_stat_dict=stats_path,
        transform=None,
        include_ki=include_ki
    )
    val_base_dataset = ZINCDataset(
        val_path,
        prop_stat_dict=stats_path,
        transform=None,
        include_ki=include_ki
    )
    test_base_dataset = ZINCDataset(
        test_path,
        prop_stat_dict=stats_path,
        transform=None,
        include_ki=include_ki
    )
    
    print(f"\nLoaded datasets:")
    print(f"  - Train: {len(train_base_dataset)} molecules")
    print(f"  - Val: {len(val_base_dataset)} molecules")
    print(f"  - Test: {len(test_base_dataset)} molecules")
    print(f"  - Total: {len(train_base_dataset) + len(val_base_dataset) + len(test_base_dataset)} molecules")
    print(f"Node features: {train_base_dataset.num_node_features}")
    print(f"Properties: {train_base_dataset.num_classes}")

    # Create fixed scattering transform
    print(f"\nScattering configuration:")
    print(f"  - Wavelet scales (J): {scattering_cfg['J']}")
    print(f"  - Moments: {scattering_cfg['num_moments']}")

    scattering_transform = FixedScatteringTransform(
        in_channels=train_base_dataset.num_node_features,
        J=scattering_cfg['J'],
        num_moments=scattering_cfg['num_moments'],
    )
    scattering_dim = scattering_transform.out_shape()
    print(f"  - Output dimension: {scattering_dim}")

    # Pre-compute scattering coefficients for each split
    print("\nPre-computing scattering coefficients...")
    train_dataset = ScatteringDataset(train_base_dataset, scattering_transform, show_progress=True)
    val_dataset = ScatteringDataset(val_base_dataset, scattering_transform, show_progress=True)
    test_dataset = ScatteringDataset(test_base_dataset, scattering_transform, show_progress=True)

    print(f"\nDataset sizes after scattering:")
    print(f"  - Train: {len(train_dataset)}")
    print(f"  - Val: {len(val_dataset)}")
    print(f"  - Test: {len(test_dataset)}")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        num_workers=training_cfg['num_workers']
    )
    valid_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=False,
        num_workers=training_cfg['num_workers']
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=training_cfg['batch_size'],
        shuffle=False,
        num_workers=training_cfg['num_workers']
    )

    # Setup logging directory
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%H-%M-%S")
    reg_str = 'regress' if reg else 'noregress'
    kl_str = 'kld' if kl_div else 'nokld'
    save_dir = os.path.join(
        logging_cfg['save_dir'],
        f"{dataset_cfg['name']}_fixed_{reg_str}_{kl_str}_{date_suffix}/"
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
            name=f"{dataset_cfg['name']}_fixed_{reg_str}_{kl_str}",
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

    # Get input dimensions from dataset
    input_dim = len(train_dataset[0][0])
    num_properties = len(train_dataset[0][1])
    len_epoch = len(train_loader)

    print(f"\nModel dimensions:")
    print(f"  - Input dim: {input_dim}")
    print(f"  - Bottleneck dim: {model_cfg['bottle_dim']}")
    print(f"  - Hidden dim: {model_cfg['hidden_dim']}")
    print(f"  - Num properties: {num_properties}")
    print(f"  - Steps per epoch: {len_epoch}")

    # Create hparams and model
    hparams = config_to_hparams(config, input_dim, num_properties, len_epoch)
    hparams.alpha = alpha
    hparams.beta = beta

    model = GRASSY(hparams=hparams)

    # Log hyperparameters
    if logger:
        logger.log_hyperparams({
            'config': config,
            'input_dim': input_dim,
            'num_properties': num_properties,
            'grassy_version': grassy_version,
            'scattering_J': scattering_cfg['J'],
            'scattering_moments': scattering_cfg['num_moments'],
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
    print("\nStarting training...")
    resume_path = config.get('resume_from_checkpoint')
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
        ckpt_path=resume_path,
    )

    # Save losses
    print("\nSaving loss histories...")
    with torch.no_grad():
        loss = model.get_loss_list()

    loss = np.array(loss)
    prefix = f"{dataset_cfg['name']}_{reg_str}_{kl_str}"

    np.save(os.path.join(save_dir, f"{prefix}_total_loss_list.npy"), loss)
    np.save(os.path.join(save_dir, f"{prefix}_recon_loss_list.npy"), np.array(model.get_recon_loss_list()))
    np.save(os.path.join(save_dir, f"{prefix}_reg_loss_list.npy"), np.array(model.get_reg_loss_list()))
    np.save(os.path.join(save_dir, f"{prefix}_kl_loss_list.npy"), np.array(model.get_kl_loss_list()))

    # Save best model
    print("\nSaving model...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        best_model = GRASSY.load_from_checkpoint(best_model_path, hparams=hparams)
        model = best_model.cpu()
        model.dev_type = 'cpu'
        torch.save(model.state_dict(), os.path.join(save_dir, f"{prefix}_model.npy"))
        print(f"Saved best model from: {best_model_path}")
    else:
        model = model.cpu()
        model.dev_type = 'cpu'
        torch.save(model.state_dict(), os.path.join(save_dir, f"{prefix}_model.npy"))
        print("Saved final model")

    # Save model to wandb
    if logger:
        logger.experiment.save(os.path.join(save_dir, f"{prefix}_model.npy"))

    # Generate and save embeddings for all splits
    print("\nGenerating embeddings for all datasets...")
    
    all_datasets = {
        'train': (train_dataset, train_base_dataset),
        'val': (val_dataset, val_base_dataset),
        'test': (test_dataset, test_base_dataset),
    }
    
    for split_name, (scattering_dataset, base_dataset) in all_datasets.items():
        print(f"\nProcessing {split_name} split...")
        
        scat_mom_list = []
        prop = [[] for _ in range(base_dataset.num_classes)]

        atom_percentage = []
        carbon = []
        nitro = []
        oxy = []
        atom_percentage.append(carbon)
        atom_percentage.append(nitro)
        atom_percentage.append(oxy)

        for index, entry in enumerate(tqdm(scattering_dataset, desc=f"Processing {split_name}")):
            scat_mom_list.append(entry[0].detach().cpu().numpy())

            # Save all properties
            for i in range(len(entry[1])):
                prop[i].append(entry[1][i].item() if torch.is_tensor(entry[1][i]) else entry[1][i])

            data = base_dataset[index]

            c = 0
            n = 0
            o = 0
            atom_count = 0
            for atom in data.element:
                if atom == 'C':
                    c += 1
                if atom == 'N':
                    n += 1
                if atom == 'O':
                    o += 1
                atom_count += 1

            c = c / atom_count if atom_count > 0 else 0
            n = n / atom_count if atom_count > 0 else 0
            o = o / atom_count if atom_count > 0 else 0
            carbon.append(c)
            nitro.append(n)
            oxy.append(o)

        scat_mom_list = np.array(scat_mom_list)

        moments = torch.Tensor(scat_mom_list)
        with torch.no_grad():
            ordered_embed = model.embed(moments)[0]

        print(f"Saving {split_name} embeddings...")
        np.save(os.path.join(save_dir, f"ordered_embedding_{prefix}_{split_name}.npy"), ordered_embed.cpu().detach().numpy())
        np.save(os.path.join(save_dir, f"scattering_coeffs_{prefix}_{split_name}.npy"), scat_mom_list)
        np.save(os.path.join(save_dir, f"embedding_prop_lists_{prefix}_{split_name}.npy"), prop)
        np.save(os.path.join(save_dir, f"atom_percentages_{prefix}_{split_name}.npy"), atom_percentage)

    print(f"\nTraining complete! Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
