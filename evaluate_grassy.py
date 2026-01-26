"""
GRASSY Evaluation Script

This script evaluates a trained GRASSY model on test data.
It computes reconstruction metrics, regression metrics (if applicable),
and generates embeddings for visualization.

Usage:
    python evaluate_grassy.py --checkpoint outputs/BACE_fixed_.../best-epoch-*.ckpt --config outputs/BACE_fixed_.../config.yaml

    python -m notebooks.evaluate_grassy --checkpoint outputs/BACE_fixed_regress_nokld_2026-01-22-21-16-05/best-epoch=98-val_loss=0.138.ckpt --config outputs/BACE_fixed_regress_nokld_2026-01-22-21-16-05/config.yaml

"""

import os
import argparse
from types import SimpleNamespace

import yaml
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt

from models.GRASSY_model import GRASSY
from models.ScatteringTransform import GraphScatteringTransform
from datasets.ZINCDataset import ZINCDataset

from utils.config_utils import load_config, apply_overrides, config_to_hparams, get_grassy_flags

class PrecomputedScatteringDataset(torch.utils.data.Dataset):
    """Load precomputed scattering coefficients."""
    
    def __init__(self, scattering_path, base_dataset):
        """
        Args:
            scattering_path: Path to scattering_moments.npy
            base_dataset: Original ZINCDataset (for properties)
        """
        self.coefficients = torch.from_numpy(np.load(scattering_path)).float()
        
        # Extract properties from base dataset
        self.properties = []
        for i in range(len(base_dataset)):
            y = base_dataset[i].y
            y = y.squeeze(0).float()  # [1, n_props] -> [n_props]
            self.properties.append(y)
        
        assert len(self.coefficients) == len(self.properties), \
            f"Mismatch: {len(self.coefficients)} coefficients vs {len(self.properties)} molecules"
        
        print(f"Loaded {len(self.coefficients)} precomputed scattering coefficients")
        print(f"Scattering dimension: {self.coefficients.shape[1]}")
    
    def __len__(self):
        return len(self.coefficients)
    
    def __getitem__(self, idx):
        return self.coefficients[idx], self.properties[idx]


class FixedScatteringTransform:
    """
    Transform that applies fixed GraphScatteringTransform to PyG Data objects.
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
    """

    def __init__(self, base_dataset, scattering_transform, show_progress=True, desc="Computing scattering"):
        self.coefficients = []
        self.properties = []

        iterator = tqdm(base_dataset, desc=desc) if show_progress else base_dataset

        for data in iterator:
            coeffs, props = scattering_transform(data)
            self.coefficients.append(coeffs)
            self.properties.append(props)

    def __len__(self):
        return len(self.coefficients)

    def __getitem__(self, idx):
        return self.coefficients[idx], self.properties[idx]

def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, property_names: list = None) -> dict:
    """
    Compute regression metrics for each property.
    
    Args:
        y_true: Ground truth values, shape (n_samples, n_properties)
        y_pred: Predicted values, shape (n_samples, n_properties)
        property_names: Optional list of property names
        
    Returns:
        Dictionary of metrics per property
    """
    n_properties = y_true.shape[1] if len(y_true.shape) > 1 else 1
    
    if n_properties == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    if property_names is None:
        property_names = [f"property_{i}" for i in range(n_properties)]
    
    metrics = {}
    
    for i, name in enumerate(property_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        
        # Skip if all values are the same (can't compute R2)
        if np.std(y_t) < 1e-8:
            continue
            
        metrics[name] = {
            'mse': mean_squared_error(y_t, y_p),
            'rmse': np.sqrt(mean_squared_error(y_t, y_p)),
            'mae': mean_absolute_error(y_t, y_p),
            'r2': r2_score(y_t, y_p),
        }
    
    # Compute average metrics
    if metrics:
        metrics['average'] = {
            'mse': np.mean([m['mse'] for m in metrics.values()]),
            'rmse': np.mean([m['rmse'] for m in metrics.values()]),
            'mae': np.mean([m['mae'] for m in metrics.values()]),
            'r2': np.mean([m['r2'] for m in metrics.values()]),
        }
    
    return metrics


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None, property_names: list = None) -> dict:
    """
    Compute classification metrics for each property (for binary classification tasks).
    
    Args:
        y_true: Ground truth labels, shape (n_samples, n_properties)
        y_pred: Predicted labels, shape (n_samples, n_properties)
        y_prob: Predicted probabilities (optional), shape (n_samples, n_properties)
        property_names: Optional list of property names
        
    Returns:
        Dictionary of metrics per property
    """
    n_properties = y_true.shape[1] if len(y_true.shape) > 1 else 1
    
    if n_properties == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
        if y_prob is not None:
            y_prob = y_prob.reshape(-1, 1)
    
    if property_names is None:
        property_names = [f"property_{i}" for i in range(n_properties)]
    
    metrics = {}
    
    for i, name in enumerate(property_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]
        
        metrics[name] = {
            'accuracy': accuracy_score(y_t, y_p),
            'precision': precision_score(y_t, y_p, zero_division=0),
            'recall': recall_score(y_t, y_p, zero_division=0),
            'f1': f1_score(y_t, y_p, zero_division=0),
        }
        
        # Add AUC if probabilities are provided
        if y_prob is not None:
            try:
                metrics[name]['auc_roc'] = roc_auc_score(y_t, y_prob[:, i])
            except ValueError:
                # Can happen if only one class is present
                metrics[name]['auc_roc'] = float('nan')
    
    return metrics


def compute_reconstruction_metrics(x_true: np.ndarray, x_recon: np.ndarray) -> dict:
    """
    Compute reconstruction metrics.
    
    Args:
        x_true: Original input, shape (n_samples, n_features)
        x_recon: Reconstructed input, shape (n_samples, n_features)
        
    Returns:
        Dictionary of reconstruction metrics
    """
    mse = mean_squared_error(x_true, x_recon)
    mae = mean_absolute_error(x_true, x_recon)
    
    # Compute per-sample reconstruction error
    sample_mse = np.mean((x_true - x_recon) ** 2, axis=1)
    
    return {
        'mse': mse,
        'rmse': np.sqrt(mse),
        'mae': mae,
        'sample_mse_mean': np.mean(sample_mse),
        'sample_mse_std': np.std(sample_mse),
        'sample_mse_median': np.median(sample_mse),
    }


def plot_regression_results(y_true: np.ndarray, y_pred: np.ndarray, property_names: list, save_path: str):
    """Plot predicted vs actual values for regression."""
    n_properties = y_true.shape[1] if len(y_true.shape) > 1 else 1
    
    if n_properties == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    n_cols = min(3, n_properties)
    n_rows = (n_properties + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_properties == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes[:n_properties], property_names)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=10)
        
        # Add diagonal line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        
        ax.set_xlabel(f'Actual {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name}\nR² = {r2_score(y_true[:, i], y_pred[:, i]):.4f}')
        ax.legend()
    
    # Hide unused axes
    for ax in axes[n_properties:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_reconstruction_histogram(x_true: np.ndarray, x_recon: np.ndarray, save_path: str):
    """Plot histogram of reconstruction errors."""
    sample_mse = np.mean((x_true - x_recon) ** 2, axis=1)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(sample_mse, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(sample_mse), color='r', linestyle='--', label=f'Mean: {np.mean(sample_mse):.4f}')
    ax.axvline(np.median(sample_mse), color='g', linestyle='--', label=f'Median: {np.median(sample_mse):.4f}')
    ax.set_xlabel('Reconstruction MSE')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Reconstruction Errors')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_latent_space(embeddings: np.ndarray, properties: np.ndarray, property_names: list, save_path: str):
    """Plot 2D visualization of latent space (first 2 dimensions or PCA)."""
    from sklearn.decomposition import PCA
    
    # Use PCA if embedding dimension > 2
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        x_label = f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)'
        y_label = f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)'
    else:
        embeddings_2d = embeddings
        x_label = 'Dim 1'
        y_label = 'Dim 2'
    
    n_properties = properties.shape[1] if len(properties.shape) > 1 else 1
    
    if n_properties == 1:
        properties = properties.reshape(-1, 1)
    
    n_cols = min(3, n_properties)
    n_rows = (n_properties + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    if n_properties == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes[:n_properties], property_names)):
        scatter = ax.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=properties[:, i],
            cmap='viridis',
            alpha=0.6,
            s=10
        )
        plt.colorbar(scatter, ax=ax, label=name)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'Latent Space colored by {name}')
    
    # Hide unused axes
    for ax in axes[n_properties:]:
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained GRASSY model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.ckpt file)')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file used for training')
    parser.add_argument('--test_path', type=str, default=None,
                        help='Override test dataset path from config')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results (default: same as checkpoint)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for evaluation')
    parser.add_argument('--no_plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--eval_train', action='store_true',
                        help='Also evaluate on training set')
    parser.add_argument('--eval_val', action='store_true',
                        help='Also evaluate on validation set')
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Extract config sections
    dataset_cfg = config['dataset']
    model_cfg = config['model']
    training_cfg = config['training']
    scattering_cfg = config['scattering']

    # Get GRASSY version flags
    grassy_version = training_cfg['grassy_version']
    kl_div, reg = get_grassy_flags(grassy_version)

    print(f"\n{'='*60}")
    print(f"GRASSY Model Evaluation")
    print(f"{'='*60}")
    print(f"\nGRASSY Version: {grassy_version}")
    print(f"  - Regression enabled: {reg}")
    print(f"  - KL Divergence enabled: {kl_div}")

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.checkpoint)
    
    eval_dir = os.path.join(args.output_dir, 'evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    print(f"\nOutput directory: {eval_dir}")

    # Determine which splits to evaluate
    splits_to_eval = ['test']
    if args.eval_train:
        splits_to_eval.append('train')
    if args.eval_val:
        splits_to_eval.append('val')
    
    print(f"Evaluating on splits: {splits_to_eval}")

    # Load datasets
    stats_path = dataset_cfg.get('stats_path')
    
    datasets = {}
    base_datasets = {}
    
    # Check if using single file format (like MOSES) or split files format (like BACE)
    dataset_name = dataset_cfg.get('name', 'UNKNOWN').lower()
    use_single_file_format = 'path' in dataset_cfg and 'train_path' not in dataset_cfg
    
    if use_single_file_format:
        # Single file format with train_size and val_size (MOSES-like)
        print(f"\nDetected single-file dataset format")
        dataset_path = dataset_cfg.get('path')
        train_size = dataset_cfg.get('train_size')
        val_size = dataset_cfg.get('val_size')
        
        if not dataset_path:
            raise ValueError("Dataset config missing 'path' for single-file format")
        
        print(f"Loading dataset: {dataset_path}")
        full_dataset = ZINCDataset(
            dataset_path,
            prop_stat_dict=stats_path,
            transform=None
        )
        total_size = len(full_dataset)
        print(f"  Total: {total_size} molecules")
        
        # Calculate splits
        test_size = total_size - train_size - val_size
        
        # Create split indices
        train_start, train_end = 0, train_size
        val_start, val_end = train_size, train_size + val_size
        test_start, test_end = train_size + val_size, total_size
        
        split_ranges = {
            'train': (train_start, train_end),
            'val': (val_start, val_end),
            'test': (test_start, test_end),
        }
        
        for split in splits_to_eval:
            if split in split_ranges:
                start, end = split_ranges[split]
                indices = list(range(start, end))
                base_datasets[split] = torch.utils.data.Subset(full_dataset, indices)
                print(f"  {split}: {len(base_datasets[split])} molecules")
    else:
        # Multi-file format with separate paths (BACE-like)
        print(f"\nDetected multi-file dataset format")
        for split in splits_to_eval:
            if split == 'test' and args.test_path:
                path = args.test_path
            else:
                path = dataset_cfg.get(f'{split}_path')
            
            if path is None:
                print(f"Warning: No path found for {split} split, skipping...")
                continue
                
            print(f"\nLoading {split} dataset: {path}")
            base_datasets[split] = ZINCDataset(
                path,
                prop_stat_dict=stats_path,
                transform=None
            )
            print(f"  Loaded {len(base_datasets[split])} molecules")

    if not base_datasets:
        raise ValueError("No datasets loaded! Check your config paths.")

    # Get dataset info from first loaded dataset
    first_dataset = next(iter(base_datasets.values()))
    if isinstance(first_dataset, torch.utils.data.Subset):
        # For Subset, get the underlying dataset
        actual_dataset = first_dataset.dataset
    else:
        actual_dataset = first_dataset
    
    num_node_features = actual_dataset.num_node_features
    num_properties = actual_dataset.num_classes
    
    print(f"\nDataset info:")
    print(f"  Node features: {num_node_features}")
    print(f"  Properties: {num_properties}")

    # Check if precomputed scattering coefficients are available
    precomputed_path = scattering_cfg.get('precomputed_path')
    use_precomputed = precomputed_path and os.path.exists(precomputed_path)
    
    print(f"\nScattering configuration:")
    print(f"  - J: {scattering_cfg['J']}")
    print(f"  - Moments: {scattering_cfg['num_moments']}")
    print(f"  - Precomputed path: {precomputed_path}")
    print(f"  - Using precomputed: {use_precomputed}")

    # Load or compute scattering coefficients
    if use_precomputed:
        print("\nLoading precomputed scattering coefficients...")
        # For precomputed scattering, we need to use the full dataset to map indices
        full_dataset = ZINCDataset(
            dataset_cfg['path'] if use_single_file_format else dataset_cfg.get('train_path'),
            prop_stat_dict=stats_path,
            transform=None
        )
        
        precomputed_dataset = PrecomputedScatteringDataset(
            scattering_path=precomputed_path,
            base_dataset=full_dataset
        )
        
        # Create subset datasets from precomputed data
        if use_single_file_format:
            # For MOSES-like format, create subsets based on split ranges
            for split, base_dataset in base_datasets.items():
                # Get indices from the subset
                if isinstance(base_dataset, torch.utils.data.Subset):
                    indices = base_dataset.indices
                else:
                    indices = list(range(len(base_dataset)))
                
                datasets[split] = torch.utils.data.Subset(precomputed_dataset, indices)
                print(f"  {split}: {len(datasets[split])} molecules")
        else:
            # For BACE-like format, load precomputed for each split
            # (would need split-specific precomputed paths for this to work)
            datasets = base_datasets
            print("Warning: Precomputed scattering not fully supported for multi-file format")
    else:
        # Compute scattering coefficients on the fly
        print("\nComputing scattering coefficients...")
        scattering_transform = FixedScatteringTransform(
            in_channels=num_node_features,
            J=scattering_cfg['J'],
            num_moments=scattering_cfg['num_moments'],
        )
        scattering_dim = scattering_transform.out_shape()
        print(f"  - Output dimension: {scattering_dim}")

        for split, base_dataset in base_datasets.items():
            datasets[split] = ScatteringDataset(
                base_dataset, 
                scattering_transform, 
                show_progress=True, 
                desc=f"Computing {split} scattering"
            )

    # Get input dimensions
    input_dim = len(datasets[next(iter(datasets.keys()))][0][0])
    
    # Create hparams and load model
    print(f"\nLoading model from: {args.checkpoint}")
    hparams = config_to_hparams(config, input_dim, num_properties,len_epoch=1)
    
    # Adjust alpha and beta based on GRASSY version
    hparams.alpha = training_cfg['alpha'] if reg else 0
    hparams.beta = training_cfg['beta'] if kl_div else 0

    model = GRASSY.load_from_checkpoint(args.checkpoint, hparams=hparams)
    model.eval()
    model.cpu()
    
    print(f"Model loaded successfully")
    print(f"  - Input dim: {input_dim}")
    print(f"  - Bottleneck dim: {model_cfg['bottle_dim']}")
    print(f"  - Hidden dim: {model_cfg['hidden_dim']}")

    # Evaluate on each split
    all_results = {}
    
    for split, dataset in datasets.items():
        print(f"\n{'='*60}")
        print(f"Evaluating on {split.upper()} set ({len(dataset)} samples)")
        print(f"{'='*60}")

        # Create data loader
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )

        # Collect predictions
        all_inputs = []
        all_reconstructions = []
        all_embeddings = []
        all_properties_true = []
        all_properties_pred = []  # all properties as regression

        with torch.no_grad():
            for batch_inputs, batch_properties in tqdm(loader, desc=f"Evaluating {split}"):
                batch_inputs = batch_inputs.float()
                
                # Forward pass returns: x_hat, y_full, mu, logvar, z
                # y_full includes all properties with num_atoms as regression
                x_hat, y_full, mu, logvar, z = model(batch_inputs)
                
                # Round num_atoms (last property) to whole numbers
                y_full_rounded = y_full.clone()
                y_full_rounded[:, -1] = torch.round(y_full[:, -1])
                
                # Use mu for deterministic embedding (no sampling)
                embeddings = mu

                all_inputs.append(batch_inputs.numpy())
                all_reconstructions.append(x_hat.numpy())
                all_embeddings.append(embeddings.numpy())
                all_properties_true.append(batch_properties.numpy())
                all_properties_pred.append(y_full_rounded.numpy())

        # Concatenate results
        all_inputs = np.concatenate(all_inputs, axis=0)
        all_reconstructions = np.concatenate(all_reconstructions, axis=0)
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_properties_true = np.concatenate(all_properties_true, axis=0)
        all_properties_pred = np.concatenate(all_properties_pred, axis=0)

        # Compute metrics
        results = {'split': split, 'n_samples': len(dataset)}

        # Reconstruction metrics
        print("\nReconstruction Metrics:")
        recon_metrics = compute_reconstruction_metrics(all_inputs, all_reconstructions)
        results['reconstruction'] = recon_metrics
        for key, value in recon_metrics.items():
            print(f"  {key}: {value:.6f}")

        # Regression metrics for all properties (including num_atoms)
        if reg:
            print("\nRegression Metrics:")
            property_names = [f"property_{i}" for i in range(num_properties)]
            reg_metrics = compute_regression_metrics(
                all_properties_true,
                all_properties_pred,
                property_names
            )
            results['regression'] = reg_metrics
            
            for prop_name, metrics in reg_metrics.items():
                print(f"\n  {prop_name}:")
                for key, value in metrics.items():
                    print(f"    {key}: {value:.6f}")

        all_results[split] = results

        # Save embeddings and predictions
        print(f"\nSaving results for {split}...")
        np.save(os.path.join(eval_dir, f'{split}_embeddings.npy'), all_embeddings)
        np.save(os.path.join(eval_dir, f'{split}_reconstructions.npy'), all_reconstructions)
        np.save(os.path.join(eval_dir, f'{split}_properties_true.npy'), all_properties_true)
        np.save(os.path.join(eval_dir, f'{split}_properties_pred.npy'), all_properties_pred)

        # Generate plots
        if not args.no_plots:
            print(f"Generating plots for {split}...")
            
            # Reconstruction error histogram
            plot_reconstruction_histogram(
                all_inputs, 
                all_reconstructions,
                os.path.join(eval_dir, f'{split}_reconstruction_histogram.png')
            )
            
            # Regression plots (all properties)
            if reg:
                property_names = [f"property_{i}" for i in range(num_properties)]
                plot_regression_results(
                    all_properties_true,
                    all_properties_pred,
                    property_names,
                    os.path.join(eval_dir, f'{split}_regression_scatter.png')
                )
            
            # Latent space visualization
            property_names = [f"property_{i}" for i in range(num_properties)]
            plot_latent_space(
                all_embeddings,
                all_properties_true,
                property_names,
                os.path.join(eval_dir, f'{split}_latent_space.png')
            )

    # Save all results to YAML
    results_path = os.path.join(eval_dir, 'evaluation_results.yaml')
    
    # Convert numpy types to Python types for YAML serialization
    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    all_results = convert_to_python_types(all_results)
    
    with open(results_path, 'w') as f:
        yaml.dump(all_results, f, default_flow_style=False)
    print(f"\nResults saved to: {results_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    for split, results in all_results.items():
        print(f"\n{split.upper()} ({results['n_samples']} samples):")
        print(f"  Reconstruction RMSE: {results['reconstruction']['rmse']:.6f}")
        if 'regression' in results and 'average' in results['regression']:
            print(f"  Regression R² (avg): {results['regression']['average']['r2']:.6f}")
            print(f"  Regression RMSE (avg): {results['regression']['average']['rmse']:.6f}")
        if 'num_atoms' in results:
            nm = results['num_atoms']['num_atoms']
            print(f"  Num_atoms acc: {nm['accuracy']:.6f}")

    print(f"\nEvaluation complete! Results saved to: {eval_dir}")


if __name__ == '__main__':
    main()