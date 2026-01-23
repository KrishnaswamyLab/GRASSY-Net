#!/usr/bin/env python3
"""
Evaluate GRASSY Model Property Prediction Quality

This script evaluates the quality of property predictions from a trained GRASSY model.
It computes various metrics (MAE, RMSE, R², Pearson correlation) per property and overall,
and generates diagnostic visualizations.

Usage:
    python evaluate_property_prediction.py \
        --model_path path/to/model.pt \
        --dataset_path path/to/dataset.npy \
        --stats_path path/to/stats.npy \
        --scatter_model path/to/scatter_model.npy \
        --output_dir results/evaluation \
        [--test_start 11000] \
        [--prop_names qed,MolWt,...]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy import stats as scipy_stats
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate GRASSY model property prediction quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required paths
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to trained model checkpoint (.pt or .npy)"
    )
    parser.add_argument(
        "--dataset_path", type=str, required=True,
        help="Path to dataset file (e.g., ZINC12K.npy)"
    )
    parser.add_argument(
        "--stats_path", type=str, required=True,
        help="Path to property statistics file (e.g., ZINC12K_stats.npy)"
    )
    parser.add_argument(
        "--scatter_model", type=str, required=True,
        help="Path to scattering transform model"
    )
    
    # Output
    parser.add_argument(
        "--output_dir", type=str, default="evaluation_results",
        help="Directory to save results and figures"
    )
    
    # Dataset split
    parser.add_argument(
        "--test_start", type=int, default=11000,
        help="Starting index for test set"
    )
    parser.add_argument(
        "--test_end", type=int, default=None,
        help="Ending index for test set (None = end of dataset)"
    )
    
    # Model hyperparameters (for reconstruction)
    parser.add_argument(
        "--bottle_dim", type=int, default=25,
        help="Bottleneck dimension"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=100,
        help="Hidden layer dimension"
    )
    parser.add_argument(
        "--alpha", type=float, default=0.3,
        help="Regression weight (alpha parameter)"
    )
    
    # Property names
    parser.add_argument(
        "--prop_names", type=str, default=None,
        help="Comma-separated property names (or path to JSON file with names)"
    )
    
    # Options
    parser.add_argument(
        "--batch_size", type=int, default=100,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--no_plots", action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device to use (auto, cpu, cuda, cuda:0, etc.)"
    )
    
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Determine the device to use."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_property_names(prop_names_arg: Optional[str], num_props: int) -> list:
    """Load or generate property names."""
    if prop_names_arg is None:
        # Default ZINC12K property names
        default_names = [
            'qed', 'HeavyAtomMolWt', 'MolWt', 'BalabanJ', 'BertzCT', 'Ipc',
            'TPSA', 'NumHAcceptors', 'NumHDonors', 'RingCount', 'MolLogP', 
            'SAscore', 'FSP3'
        ]
        if num_props <= len(default_names):
            return default_names[:num_props]
        return default_names + [f"Prop_{i}" for i in range(len(default_names), num_props)]
    
    # Check if it's a file path
    if Path(prop_names_arg).exists():
        with open(prop_names_arg, 'r') as f:
            return json.load(f)
    
    # Otherwise, parse as comma-separated
    return [name.strip() for name in prop_names_arg.split(',')]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics for predictions.
    
    Args:
        y_true: Ground truth values [N, num_props] or [N]
        y_pred: Predicted values [N, num_props] or [N]
    
    Returns:
        Dictionary of metrics
    """
    # Ensure 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)
    
    n_samples, n_props = y_true.shape
    
    metrics = {
        'per_property': {},
        'overall': {}
    }
    
    all_mae = []
    all_rmse = []
    all_r2 = []
    all_pearson = []
    
    for i in range(n_props):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        
        # Mean Absolute Error
        mae = np.mean(np.abs(yt - yp))
        
        # Root Mean Squared Error
        rmse = np.sqrt(np.mean((yt - yp) ** 2))
        
        # R² Score
        ss_res = np.sum((yt - yp) ** 2)
        ss_tot = np.sum((yt - np.mean(yt)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        # Pearson Correlation
        if np.std(yt) > 0 and np.std(yp) > 0:
            pearson_r, pearson_p = scipy_stats.pearsonr(yt, yp)
        else:
            pearson_r, pearson_p = 0.0, 1.0
        
        # Value ranges
        true_min, true_max = yt.min(), yt.max()
        pred_min, pred_max = yp.min(), yp.max()
        
        # Normalized MAE (as percentage of range)
        value_range = true_max - true_min
        nmae = (mae / value_range * 100) if value_range > 0 else 0.0
        
        metrics['per_property'][i] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'nmae_percent': float(nmae),
            'true_range': [float(true_min), float(true_max)],
            'pred_range': [float(pred_min), float(pred_max)],
            'true_mean': float(np.mean(yt)),
            'true_std': float(np.std(yt)),
            'pred_mean': float(np.mean(yp)),
            'pred_std': float(np.std(yp))
        }
        
        all_mae.append(mae)
        all_rmse.append(rmse)
        all_r2.append(r2)
        all_pearson.append(pearson_r)
    
    # Overall metrics (averaged across properties)
    metrics['overall'] = {
        'mean_mae': float(np.mean(all_mae)),
        'std_mae': float(np.std(all_mae)),
        'mean_rmse': float(np.mean(all_rmse)),
        'std_rmse': float(np.std(all_rmse)),
        'mean_r2': float(np.mean(all_r2)),
        'std_r2': float(np.std(all_r2)),
        'mean_pearson': float(np.mean(all_pearson)),
        'std_pearson': float(np.std(all_pearson)),
        'n_samples': n_samples,
        'n_properties': n_props
    }
    
    return metrics


def generate_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prop_names: list,
    metrics: dict,
    output_dir: Path
):
    """Generate diagnostic visualization plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
    except ImportError:
        print("Warning: matplotlib not available, skipping plots")
        return
    
    n_props = y_true.shape[1]
    
    # 1. Parity plots (predicted vs true) for each property
    n_cols = min(4, n_props)
    n_rows = (n_props + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = np.atleast_2d(axes)
    
    for i in range(n_props):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[0, col]
        
        yt = y_true[:, i]
        yp = y_pred[:, i]
        
        ax.scatter(yt, yp, alpha=0.3, s=10, c='steelblue')
        
        # Perfect prediction line
        lims = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        ax.plot(lims, lims, 'r--', lw=1.5, label='Perfect')
        
        # Add metrics annotation
        m = metrics['per_property'][i]
        ax.text(0.05, 0.95, f"R²={m['r2']:.3f}\nMAE={m['mae']:.4f}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'True {prop_names[i]}')
        ax.set_ylabel(f'Predicted {prop_names[i]}')
        ax.set_title(prop_names[i])
    
    # Hide empty subplots
    for i in range(n_props, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[0, col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parity_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Error distribution plots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = np.atleast_2d(axes)
    
    for i in range(n_props):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[0, col]
        
        errors = y_pred[:, i] - y_true[:, i]
        
        ax.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', lw=1.5)
        ax.axvline(errors.mean(), color='orange', linestyle='-', lw=1.5, label=f'Mean={errors.mean():.4f}')
        
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Count')
        ax.set_title(f'{prop_names[i]} Error Distribution')
        ax.legend(fontsize=8)
    
    for i in range(n_props, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[0, col]
        ax.set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Summary bar chart of metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    x = np.arange(n_props)
    width = 0.6
    
    # MAE
    maes = [metrics['per_property'][i]['mae'] for i in range(n_props)]
    axes[0].bar(x, maes, width, color='steelblue', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(prop_names, rotation=45, ha='right')
    axes[0].set_ylabel('MAE')
    axes[0].set_title('Mean Absolute Error by Property')
    axes[0].axhline(np.mean(maes), color='red', linestyle='--', label=f'Mean={np.mean(maes):.4f}')
    axes[0].legend()
    
    # R²
    r2s = [metrics['per_property'][i]['r2'] for i in range(n_props)]
    colors = ['green' if r > 0.8 else 'orange' if r > 0.5 else 'red' for r in r2s]
    axes[1].bar(x, r2s, width, color=colors, alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(prop_names, rotation=45, ha='right')
    axes[1].set_ylabel('R²')
    axes[1].set_title('R² Score by Property')
    axes[1].axhline(0.8, color='green', linestyle=':', alpha=0.5)
    axes[1].axhline(0.5, color='orange', linestyle=':', alpha=0.5)
    axes[1].set_ylim(0, 1.05)
    
    # Pearson correlation
    pearsons = [metrics['per_property'][i]['pearson_r'] for i in range(n_props)]
    axes[2].bar(x, pearsons, width, color='purple', alpha=0.8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(prop_names, rotation=45, ha='right')
    axes[2].set_ylabel('Pearson r')
    axes[2].set_title('Pearson Correlation by Property')
    axes[2].set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Normalized MAE (as % of range) - useful for comparing across properties
    fig, ax = plt.subplots(figsize=(10, 5))
    
    nmaes = [metrics['per_property'][i]['nmae_percent'] for i in range(n_props)]
    colors = ['green' if n < 5 else 'orange' if n < 10 else 'red' for n in nmaes]
    bars = ax.bar(x, nmaes, width, color=colors, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(prop_names, rotation=45, ha='right')
    ax.set_ylabel('Normalized MAE (%)')
    ax.set_title('Normalized MAE (% of value range) by Property')
    ax.axhline(5, color='green', linestyle=':', alpha=0.5, label='5% threshold')
    ax.axhline(10, color='orange', linestyle=':', alpha=0.5, label='10% threshold')
    ax.legend()
    
    # Add value labels on bars
    for bar, nmae in zip(bars, nmaes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{nmae:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'normalized_mae.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved plots to {output_dir}")


def print_results_table(metrics: dict, prop_names: list):
    """Print a formatted results table."""
    n_props = len(metrics['per_property'])
    
    print("\n" + "=" * 90)
    print("PROPERTY PREDICTION EVALUATION RESULTS")
    print("=" * 90)
    
    # Header
    header = f"{'Property':<18} | {'MAE':<10} | {'RMSE':<10} | {'R²':<8} | {'Pearson r':<10} | {'NMAE %':<8}"
    print(header)
    print("-" * 90)
    
    # Per-property results
    for i in range(n_props):
        m = metrics['per_property'][i]
        name = prop_names[i] if i < len(prop_names) else f"Prop_{i}"
        
        # Color coding for R² (using ANSI codes)
        r2_val = m['r2']
        if r2_val >= 0.9:
            r2_str = f"\033[92m{r2_val:.4f}\033[0m"  # Green
        elif r2_val >= 0.7:
            r2_str = f"\033[93m{r2_val:.4f}\033[0m"  # Yellow
        else:
            r2_str = f"\033[91m{r2_val:.4f}\033[0m"  # Red
        
        # Plain version without colors (for file output)
        row = f"{name:<18} | {m['mae']:<10.6f} | {m['rmse']:<10.6f} | {m['r2']:<8.4f} | {m['pearson_r']:<10.4f} | {m['nmae_percent']:<8.2f}"
        print(row)
    
    print("-" * 90)
    
    # Overall summary
    o = metrics['overall']
    print(f"\n{'OVERALL SUMMARY':^90}")
    print("-" * 90)
    print(f"  Number of test samples: {o['n_samples']}")
    print(f"  Number of properties:   {o['n_properties']}")
    print(f"  Mean MAE:     {o['mean_mae']:.6f} ± {o['std_mae']:.6f}")
    print(f"  Mean RMSE:    {o['mean_rmse']:.6f} ± {o['std_rmse']:.6f}")
    print(f"  Mean R²:      {o['mean_r2']:.4f} ± {o['std_r2']:.4f}")
    print(f"  Mean Pearson: {o['mean_pearson']:.4f} ± {o['std_pearson']:.4f}")
    print("=" * 90)


def main():
    args = parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Try to import the model and dataset classes
    # We'll try multiple import strategies
    print("\nLoading model and dataset...")
    
    try:
        # Strategy 1: Direct import (if running from project root)
        from models.GRASSY_model import GRASSY
        from datasets.ZINCDataset import ZINCDataset, Scattering
        print("  Imported from models/datasets packages")
    except ImportError:
        try:
            # Strategy 2: Add parent directory to path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from models.GRASSY_model import GRASSY
            from datasets.ZINCDataset import ZINCDataset, Scattering
            print("  Imported from parent directory")
        except ImportError:
            print("\nERROR: Could not import GRASSY model and dataset classes.")
            print("Please ensure you're running from the project root directory,")
            print("or that the 'models' and 'datasets' packages are in your PYTHONPATH.")
            print("\nAlternatively, provide the paths as environment variables or")
            print("modify this script to point to your specific module locations.")
            sys.exit(1)
    
    # Load dataset
    print(f"  Loading dataset from: {args.dataset_path}")
    dataset = ZINCDataset(
        args.dataset_path,
        prop_stat_dict=args.stats_path,
        transform=Scattering(scatter_model_name=args.scatter_model)
    )
    
    # Determine test indices
    test_start = args.test_start
    test_end = args.test_end if args.test_end else len(dataset)
    test_indices = list(range(test_start, test_end))
    print(f"  Test set: indices {test_start} to {test_end} ({len(test_indices)} samples)")
    
    # Get a sample to determine dimensions
    sample_x, sample_y = dataset[0]
    input_dim = len(sample_x)
    num_properties = len(sample_y)
    print(f"  Input dimension: {input_dim}")
    print(f"  Number of properties: {num_properties}")
    
    # Load property names
    prop_names = load_property_names(args.prop_names, num_properties)
    print(f"  Properties: {', '.join(prop_names)}")
    
    # Reconstruct model architecture
    from argparse import Namespace as NS
    hparams = NS(
        input_dim=input_dim,
        bottle_dim=args.bottle_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=0.001,
        alpha=args.alpha,
        beta=0.0,
        n_epochs=100,
        len_epoch=None,
        batch_size=args.batch_size,
        n_gpus=0,
        num_properties=num_properties
    )
    
    model = GRASSY(hparams=hparams)
    
    # Load model weights
    print(f"  Loading model from: {args.model_path}")
    model_state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    # Collect predictions
    print("\nRunning evaluation...")
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for idx in tqdm(test_indices, desc="Evaluating"):
            x, y = dataset[idx]
            x_t = x.unsqueeze(0).float().to(device)
            
            # Model forward pass
            # GRASSY returns: (z, y_hat, mu, logvar, x_recon) or similar
            outputs = model(x_t)
            y_hat = outputs[1]  # Assuming y_hat is the second output
            
            all_true.append(np.array(y))
            all_pred.append(y_hat.cpu().numpy().squeeze())
    
    # Convert to arrays
    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    
    print(f"  Collected {len(y_true)} predictions")
    print(f"  True shape: {y_true.shape}, Pred shape: {y_pred.shape}")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(y_true, y_pred)
    
    # Print results table
    print_results_table(metrics, prop_names)
    
    # Save metrics to JSON
    metrics_file = output_dir / 'metrics.json'
    metrics['property_names'] = prop_names
    metrics['config'] = {
        'model_path': args.model_path,
        'dataset_path': args.dataset_path,
        'test_start': test_start,
        'test_end': test_end,
        'alpha': args.alpha,
        'bottle_dim': args.bottle_dim,
        'hidden_dim': args.hidden_dim
    }
    
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_file}")
    
    # Save predictions
    np.savez(
        output_dir / 'predictions.npz',
        y_true=y_true,
        y_pred=y_pred,
        property_names=prop_names
    )
    print(f"Saved predictions to: {output_dir / 'predictions.npz'}")
    
    # Generate plots
    if not args.no_plots:
        print("\nGenerating plots...")
        generate_plots(y_true, y_pred, prop_names, metrics, output_dir)
    
    print("\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_dir}")
    
    return metrics


if __name__ == "__main__":
    main()