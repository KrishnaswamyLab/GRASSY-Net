"""
Learnable Scattering MLP Training Script with YAML Configuration

Usage:
    python train_scattering_from_config.py                           # Use default scattering_config.yaml
    python train_scattering_from_config.py --config my_config.yaml   # Use custom config file
    python train_scattering_from_config.py --config scattering_config.yaml --override training.max_epochs=50
"""

import os
import datetime
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import trange
import wandb

from models.MLP_Scattering_module import TSNet as TSNet_MLP
from models.Scattering_module import TSNet as TSNet_Abs
from datasets.load_ZINC_tranche import ZINCDataset


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_overrides(config: dict, overrides: list) -> dict:
    """Apply command-line overrides to config.

    Example: --override training.max_epochs=50 model.trainable_scattering=true
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


def split_dataset(dataset, splits, seed):
    """Splits data into non-overlapping datasets of given proportions."""
    splits = np.array(splits)
    splits = splits / np.sum(splits)
    n = len(dataset)
    torch.manual_seed(seed)
    val_size = int(splits[1] * n)
    test_size = int(splits[2] * n)
    train_size = n - val_size - test_size
    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    return train_set, val_set, test_set


class EarlyStopping:
    """Early Stopping implementation."""

    def __init__(self, mode='min', min_delta=0, patience=8, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if metrics != metrics:  # Handle NaN
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


def accuracy(model, dataset, loss_fn, device):
    """Compute accuracy/loss on a dataset."""
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred, sc = model(data)
            total_loss += loss_fn(pred, data.y)

    acc = total_loss / len(dataset)
    return acc, pred


def evaluate(model, loss_fn, train_ds, test_ds, val_ds, device):
    """Evaluate model on all splits."""
    train_acc, train_pred = accuracy(model, train_ds, loss_fn, device)
    test_acc, test_pred = accuracy(model, test_ds, loss_fn, device)
    val_acc, val_pred = accuracy(model, val_ds, loss_fn, device)

    results = {
        "train_acc": train_acc,
        "train_pred": train_pred,
        "test_acc": test_acc,
        "test_pred": test_pred,
        "val_acc": val_acc,
        "val_pred": val_pred,
        "state_dict": model.state_dict(),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description='Train Learnable Scattering MLP with YAML configuration')
    parser.add_argument('--config', type=str, default='scattering_config.yaml',
                        help='Path to config file (default: scattering_config.yaml)')
    parser.add_argument('--override', type=str, nargs='*', default=[],
                        help='Override config values (e.g., training.max_epochs=50)')
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
    early_stopping_cfg = config.get('early_stopping', {'enabled': True})

    # Setup device
    if hardware_cfg.get('accelerator', 'auto') == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif hardware_cfg['accelerator'] == 'gpu':
        device = torch.device("cuda")
    elif hardware_cfg['accelerator'] == 'mps':
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nUsing device: {device}")

    # Load dataset
    print(f"\nLoading dataset: {dataset_cfg['name']}")
    dataset = ZINCDataset(
        dataset_cfg['path'],
        prop_stat_dict=dataset_cfg.get('stats_path'),
        include_ki=dataset_cfg.get('include_ki', False)
    )

    # Data splits
    splits = (
        dataset_cfg.get('train_split', 0.8),
        dataset_cfg.get('val_split', 0.1),
        dataset_cfg.get('test_split', 0.1)
    )
    seed = dataset_cfg.get('seed', 42)
    train_ds, val_ds, test_ds = split_dataset(dataset, splits, seed)

    print(f"Dataset splits: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    # Create data loader
    train_loader = DataLoader(
        train_ds,
        batch_size=training_cfg['batch_size'],
        shuffle=True,
        num_workers=training_cfg.get('num_workers', 0)
    )


    # Create model
    nonlinearity = model_cfg.get('nonlinearity', 'mlp')
    trainable_scattering = model_cfg.get('trainable_scattering', False)
    max_graph_size = model_cfg.get('max_graph_size', 100)

    print(f"\nModel configuration:")
    print(f"  - Nonlinearity: {nonlinearity}")
    print(f"  - Trainable scattering: {trainable_scattering}")
    print(f"  - Max graph size: {max_graph_size}")

    if nonlinearity == 'mlp':
        model = TSNet_MLP(
            dataset.num_node_features,
            dataset.num_classes,
            trainable_scattering=trainable_scattering,
            max_graph_size=max_graph_size
        )
    elif nonlinearity == 'abs':
        model = TSNet_Abs(
            dataset.num_node_features,
            dataset.num_classes,
            trainable_laziness=trainable_scattering
        )
    else:
        raise ValueError(f"Unknown nonlinearity: {nonlinearity}. Options: 'mlp', 'abs'")

    model = model.to(device)

    # Setup logging directory
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%H-%M-%S")
    save_dir = os.path.join(
        logging_cfg['save_dir'],
        f"{dataset_cfg['name']}_scattering_{date_suffix}/"
    )

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"\nSave directory: {save_dir}")

    # Save config to output directory for reproducibility
    config_save_path = os.path.join(save_dir, 'scattering_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to: {config_save_path}")

    # Initialize W&B
    if logging_cfg['wandb']['enabled']:
        wandb_cfg = logging_cfg['wandb']
        wandb.init(
            entity=wandb_cfg['entity'],
            project=wandb_cfg['project'],
            name=f"{dataset_cfg['name']}_scattering_{nonlinearity}_{date_suffix}",
            config={
                "dataset": dataset_cfg['name'],
                "batch_size": training_cfg['batch_size'],
                "learning_rate": training_cfg['learning_rate'],
                "max_epochs": training_cfg['max_epochs'],
                "early_stopping_patience": early_stopping_cfg.get('patience', 5),
                "num_node_features": dataset.num_node_features,
                "num_classes": dataset.num_classes,
                "nonlinearity": nonlinearity,
                "trainable_scattering": trainable_scattering,
                "max_graph_size": max_graph_size,
            }
        )

    # Setup training
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg['learning_rate'])
    loss_fn = torch.nn.MSELoss()

    # Early stopping
    if early_stopping_cfg.get('enabled', True):
        early_stopper = EarlyStopping(
            mode=early_stopping_cfg.get('mode', 'min'),
            patience=early_stopping_cfg.get('patience', 5),
            min_delta=early_stopping_cfg.get('min_delta', 0),
            percentage=early_stopping_cfg.get('percentage', False)
        )
        print(f"Early stopping: enabled (patience={early_stopping_cfg.get('patience', 5)})")
    else:
        early_stopper = None
        print("Early stopping: disabled")

    # Training loop
    print("\nStarting training...")
    results_compiled = []
    best_val_loss = float('inf')
    best_model_state = None

    model.train()
    max_epochs = training_cfg['max_epochs']

    for epoch in trange(1, max_epochs + 1):
        model.train()

        for data in train_loader:
            optimizer.zero_grad()
            data = data.to(device)
            out, sc = model(data)
            loss = loss_fn(out, data.y)
            loss.backward()
            optimizer.step()

            if logging_cfg['wandb']['enabled']:
                wandb.log({"train_loss_batch": loss.item()})

        # Evaluate
        results = evaluate(model, loss_fn, train_ds, test_ds, val_ds, device)

        train_loss = results['train_acc'].item() if torch.is_tensor(results['train_acc']) else results['train_acc']
        val_loss = results['val_acc'].item() if torch.is_tensor(results['val_acc']) else results['val_acc']
        test_loss = results['test_acc'].item() if torch.is_tensor(results['test_acc']) else results['test_acc']

        if logging_cfg['wandb']['enabled']:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss": test_loss,
            })

        print(f'Epoch: {epoch}, Train: {train_loss:.6f}, Val: {val_loss:.6f}, Test: {test_loss:.6f}')
        results_compiled.append(test_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.scatter.state_dict().copy()

        # Early stopping check
        if early_stopper is not None and early_stopper.step(results['val_acc']):
            print("Early stopping criterion met. Ending training.")
            if logging_cfg['wandb']['enabled']:
                wandb.log({"early_stopped": True, "stopped_at_epoch": epoch})
            break

    # Final evaluation
    model.eval()
    results = evaluate(model, loss_fn, train_ds, test_ds, val_ds, device)

    final_train_loss = results['train_acc'].item() if torch.is_tensor(results['train_acc']) else results['train_acc']
    final_val_loss = results['val_acc'].item() if torch.is_tensor(results['val_acc']) else results['val_acc']
    final_test_loss = results['test_acc'].item() if torch.is_tensor(results['test_acc']) else results['test_acc']

    if logging_cfg['wandb']['enabled']:
        wandb.log({
            "final_train_loss": final_train_loss,
            "final_val_loss": final_val_loss,
            "final_test_loss": final_test_loss,
        })

    print(f"\nFinal Results - Train: {final_train_loss:.6f}, Val: {final_val_loss:.6f}, Test: {final_test_loss:.6f}")
    print("Results compiled:", results_compiled)

    # Save model
    print('\nSaving scatter model...')
    model_save_path = os.path.join(logging_cfg['model_save_dir'], f"{dataset_cfg['name']}_scattering_{nonlinearity}_{date_suffix}.npy")

    # Ensure model save directory exists
    os.makedirs(logging_cfg['model_save_dir'], exist_ok=True)

    # Save best model if available, otherwise save final model
    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print(f"Saved best model (val_loss={best_val_loss:.6f}) to: {model_save_path}")
    else:
        torch.save(model.scatter.state_dict(), model_save_path)
        print(f"Saved final model to: {model_save_path}")

    # Also save to the run directory
    run_model_path = os.path.join(save_dir, f"{dataset_cfg['name']}_scatter.npy")
    if best_model_state is not None:
        torch.save(best_model_state, run_model_path)
    else:
        torch.save(model.scatter.state_dict(), run_model_path)

    # Save training history
    np.save(os.path.join(save_dir, f"{dataset_cfg['name']}_test_loss_history.npy"), np.array(results_compiled))

    if logging_cfg['wandb']['enabled']:
        wandb.save(model_save_path)
        wandb.finish()

    print(f"\nTraining complete! Results saved to: {save_dir}")
    print(f"Scatter model saved to: {model_save_path}")


if __name__ == '__main__':
    main()
