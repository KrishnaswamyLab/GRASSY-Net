"""
GRASSY Training Script with YAML Configuration

Usage:
    python train_grassy.py                      # Use default config.yaml
    python train_grassy.py --config my_config.yaml  # Use custom config file
    python train_grassy.py --config config.yaml --override training.n_epochs=50
"""

import os
import datetime
import argparse
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

from models.GRASSY_model import GRASSY
from datasets.load_ZINC_tranche import ZINCDataset, Scattering


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_overrides(config: dict, overrides: list) -> dict:
    """Apply command-line overrides to config.

    Example: --override training.n_epochs=50 model.hidden_dim=200
    """
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Invalid override format: {override}. Use key.subkey=value")

        key_path, value = override.split('=', 1)
        keys = key_path.split('.')

        # Navigate to the parent dict
        d = config
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]

        # Try to parse the value as the appropriate type
        final_key = keys[-1]
        try:
            # Try int
            parsed_value = int(value)
        except ValueError:
            try:
                # Try float
                parsed_value = float(value)
            except ValueError:
                # Try bool
                if value.lower() in ('true', 'false'):
                    parsed_value = value.lower() == 'true'
                elif value.lower() == 'null' or value.lower() == 'none':
                    parsed_value = None
                else:
                    # Keep as string
                    parsed_value = value

        d[final_key] = parsed_value
        print(f"Override applied: {key_path} = {parsed_value}")

    return config


def config_to_hparams(config: dict, input_dim: int, num_properties: int, len_epoch: int) -> SimpleNamespace:
    """Convert config dict to hparams namespace for GRASSY model."""
    hparams = SimpleNamespace(
        input_dim=input_dim,
        bottle_dim=config['model']['bottle_dim'],
        hidden_dim=config['model']['hidden_dim'],
        learning_rate=config['training']['learning_rate'],
        alpha=config['training']['alpha'],
        beta=config['training']['beta'],
        n_epochs=config['training']['n_epochs'],
        len_epoch=len_epoch,
        num_properties=num_properties,
        n_gpus=config['hardware']['n_gpus'],
    )
    return hparams


def get_grassy_flags(grassy_version: str) -> tuple:
    """Get kl_div and reg flags based on GRASSY version."""
    version_map = {
        'AE+REG': (False, True),
        'VAE+REG': (True, True),
        'AE': (False, False),
        'VAE': (True, False),
    }
    if grassy_version not in version_map:
        raise ValueError(f"Unknown GRASSY version: {grassy_version}. Options: {list(version_map.keys())}")
    return version_map[grassy_version]


def main():
    parser = argparse.ArgumentParser(description='Train GRASSY model with YAML configuration')
    parser.add_argument('--config', type=str, default='grassy_config.yaml',
                        help='Path to config file (default: config.yaml)')
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
    early_stopping_cfg = config.get('early_stopping', {'enabled': False})

    # Get GRASSY version flags
    grassy_version = training_cfg['grassy_version']
    kl_div, reg = get_grassy_flags(grassy_version)

    # Adjust alpha and beta based on GRASSY version
    alpha = training_cfg['alpha'] if reg else 0
    beta = training_cfg['beta'] if kl_div else 0

    print(f"\nGRASSY Version: {grassy_version}")
    print(f"  - KL Divergence: {'enabled' if kl_div else 'disabled'}")
    print(f"  - Regression: {'enabled' if reg else 'disabled'}")
    print(f"  - Alpha (reg weight): {alpha}")
    print(f"  - Beta (KL weight): {beta}")

    # Load dataset
    print(f"\nLoading dataset: {dataset_cfg['name']}")
    full_dataset = ZINCDataset(
        dataset_cfg['path'],
        prop_stat_dict=dataset_cfg['stats_path'],
        transform=Scattering(scatter_model_name=dataset_cfg['scatter_model_path'])
    )

    # Data splits
    train_size = dataset_cfg['train_size']
    val_size = dataset_cfg['val_size']
    test_size = len(full_dataset) - train_size - val_size

    print(f"Dataset splits: train={train_size}, val={val_size}, test={test_size}")

    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(dataset_cfg['seed'])
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        num_workers=training_cfg['num_workers']
    )
    valid_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=training_cfg['batch_size'],
        shuffle=False,
        num_workers=training_cfg['num_workers']
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
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
        f"{dataset_cfg['name']}_{reg_str}_{kl_str}_{date_suffix}/"
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
            name=f"{dataset_cfg['name']}_{reg_str}_{kl_str}",
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
    input_dim = len(train_set[0][0])
    num_properties = len(train_set[0][1])
    len_epoch = len(train_loader)

    print(f"\nModel dimensions:")
    print(f"  - Input dim: {input_dim}")
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

    # Generate and save embeddings
    print("\nGenerating embeddings for full dataset...")
    no_transform_dataset = ZINCDataset(dataset_cfg['path'])

    scat_mom_list = []
    prop = []
    qed = []
    heavywt = []
    tpsa = []
    ringcount = []

    prop.append(qed)
    prop.append(heavywt)
    prop.append(tpsa)
    prop.append(ringcount)

    atom_percentage = []
    carbon = []
    nitro = []
    oxy = []
    atom_percentage.append(carbon)
    atom_percentage.append(nitro)
    atom_percentage.append(oxy)

    for index, entry in enumerate(tqdm(full_dataset)):
        scat_mom_list.append(entry[0].detach().cpu().numpy())
        qed.append(entry[1][0])
        heavywt.append(entry[1][1])
        tpsa.append(entry[1][6])
        ringcount.append(entry[1][9])

        data = no_transform_dataset[index]

        c = 0
        n = 0
        o = 0
        i = 0
        for atom in data.element:
            if atom == 'C':
                c += 1
            if atom == 'N':
                n += 1
            if atom == 'O':
                o += 1
            i += 1

        c = c / i
        n = n / i
        o = o / i
        carbon.append(c)
        nitro.append(n)
        oxy.append(o)

    scat_mom_list = np.array(scat_mom_list)

    moments = torch.Tensor(scat_mom_list)
    with torch.no_grad():
        ordered_embed = model.embed(moments)[0]

    print("Saving embeddings...")
    np.save(os.path.join(save_dir, f"ordered_embedding_{prefix}.npy"), ordered_embed.cpu().detach().numpy())
    np.save(os.path.join(save_dir, f"embedding_prop_lists_{prefix}.npy"), prop)
    np.save(os.path.join(save_dir, f"atom_percentages_{prefix}.npy"), atom_percentage)

    print(f"\nTraining complete! Results saved to: {save_dir}")


if __name__ == '__main__':
    main()
