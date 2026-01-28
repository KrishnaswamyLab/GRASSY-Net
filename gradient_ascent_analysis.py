"""
Gradient Ascent Analysis in Latent Space

This script performs gradient ascent optimization in the latent space of the GRASSY model
and visualizes the trajectories in both PCA and PHATE spaces.

Usage:
    python gradient_ascent_analysis.py --training_dir ../outputs/BACE_fixed_regress_nokld_2026-01-23-22-26-21/
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Try to import PHATE
try:
    import phate
    PHATE_AVAILABLE = True
except ImportError:
    print("Warning: PHATE not installed. Install with: pip install phate")
    PHATE_AVAILABLE = False

from models.GRASSY_model import GRASSY
from models.LatentOptimization import LatentOptimizer
from utils.config_utils import load_config, config_to_hparams


def auto_detect_files(directory):
    """Auto-detect output files from training directory."""
    files = {}
    if os.path.exists(directory):
        for f in os.listdir(directory):
            if f.endswith('.npy'):
                if 'ordered_embedding' in f:
                    files['embedding'] = os.path.join(directory, f)
                elif 'embedding_prop_lists' in f:
                    files['properties'] = os.path.join(directory, f)
                elif 'scattering_coeffs' in f:
                    files['scattering'] = os.path.join(directory, f)
                elif 'atom_percentages' in f:
                    files['atom_percentages'] = os.path.join(directory, f)
    return files


def load_data(training_dir):
    """Load embeddings, properties, and scattering coefficients."""
    print("Loading data...")
    detected_files = auto_detect_files(training_dir)
    
    # Load embeddings
    embedding_path = detected_files.get('embedding')
    if embedding_path and os.path.exists(embedding_path):
        embeddings = np.load(embedding_path)
        print(f"  Loaded embeddings: {embeddings.shape}")
    else:
        raise FileNotFoundError(f"Embedding file not found in {training_dir}")
    
    # Load properties
    properties_path = detected_files.get('properties')
    if properties_path and os.path.exists(properties_path):
        properties = np.load(properties_path, allow_pickle=True)
        if isinstance(properties, np.ndarray) and properties.dtype == object:
            properties = np.array([np.array(p) for p in properties])
        print(f"  Loaded properties: {properties.shape}")
    else:
        properties = None
        print("  Properties file not found")
    
    # Load scattering coefficients
    scattering_path = detected_files.get('scattering')
    if scattering_path and os.path.exists(scattering_path):
        scattering_coeffs = np.load(scattering_path)
        print(f"  Loaded scattering coefficients: {scattering_coeffs.shape}")
    else:
        scattering_coeffs = None
        print("  Scattering coefficients not found")
    
    # Load atom percentages
    atom_path = detected_files.get('atom_percentages')
    if atom_path and os.path.exists(atom_path):
        atom_percentages = np.load(atom_path, allow_pickle=True)
        print(f"  Loaded atom percentages: {atom_percentages.shape}")
    else:
        atom_percentages = None
    
    return embeddings, properties, scattering_coeffs, atom_percentages


def prepare_properties(properties, embeddings, num_properties=None):
    """Prepare property arrays for visualization.
    
    Note: If num_properties is provided, the last property is treated as num_atoms (discrete).
    """
    prop_arrays = []
    
    if properties is not None:
        if properties.ndim == 1:
            prop_arrays = [np.array(p) for p in properties]
        elif properties.ndim == 2:
            if properties.shape[0] == embeddings.shape[0]:
                prop_arrays = [properties[:, i] for i in range(properties.shape[1])]
            else:
                prop_arrays = [properties[i, :] for i in range(properties.shape[0])]
    
    return prop_arrays


def load_model(training_dir, scattering_dim, num_properties, device):
    """Load the trained GRASSY model from checkpoint."""
    print(f"\nLoading model from {training_dir}...")
    
    # Find checkpoint
    checkpoint_files = []
    if os.path.exists(training_dir):
        for f in os.listdir(training_dir):
            if f.endswith('.ckpt'):
                checkpoint_files.append(os.path.join(training_dir, f))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No .ckpt checkpoint files found in {training_dir}")
    
    checkpoint_path = checkpoint_files[0]
    print(f"  Found checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Load config
    config_path = os.path.join(training_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.yaml not found in {training_dir}")
    
    config = load_config(config_path)
    
    # Reconstruct hparams
    hparams = config_to_hparams(config, scattering_dim, num_properties, len_epoch=1)
    
    # Load model
    model = GRASSY.load_from_checkpoint(checkpoint_path, hparams=hparams)
    model = model.to(device)
    if hasattr(model, 'dev_type'):
        model.dev_type = device.type
    model.eval()
    
    print("  Model loaded successfully!")
    return model


def run_gradient_ascent(model, embeddings, prop_arrays, device, n_trajectories=5, n_steps=50, 
                       step_size=0.1, noise_scale=0.01):
    """Run gradient ascent optimization with multiple noisy trajectories from a single starting point."""
    print(f"\nRunning gradient ascent ({n_trajectories} noisy trajectories, {n_steps} steps, noise={noise_scale})...")
    # Select single starting point (median of first property)
    # Select starting point near center of latent space
    embeddings_center = np.mean(embeddings, axis=0)
    distances_to_center = np.linalg.norm(embeddings - embeddings_center, axis=1)
    start_idx = np.argmin(distances_to_center)
    # start_idx = np.random.randint(len(embeddings))
    
    print(f"  Selected starting point: index {start_idx}")
    
    # Run optimization with noise
    trajectories = []
    all_num_atoms = []
    for traj_num in range(n_trajectories):
        # Initialize optimizer with noise for this trajectory
        optimizer = LatentOptimizer(model=model, property_idx=0, step_size=step_size, noise_scale=noise_scale)
        
        z_start = torch.tensor(embeddings[start_idx:start_idx+1], dtype=torch.float32, device=device)
        # optimize returns: (latent_trajectory, decoded_trajectory, num_atoms_trajectory)
        latent_traj, _, num_atoms_traj = optimizer.optimize(z_start, n_steps=n_steps, return_decoded=True)
        
        latent_traj_np = [z.cpu().numpy() for z in latent_traj]
        trajectories.append(np.concatenate(latent_traj_np, axis=0))
        all_num_atoms.append(num_atoms_traj)
        print(f"    Trajectory {traj_num + 1}/{n_trajectories} complete")
    
    print("  Gradient ascent optimization complete!")
    return trajectories, all_num_atoms


def project_trajectories(trajectories, embeddings, scaler, pca, phate_op=None):
    """Project trajectories to PCA and PHATE spaces."""
    print("\nProjecting trajectories...")
    
    # Project to PCA
    trajectories_pca = []
    for traj in trajectories:
        traj_scaled = scaler.transform(traj)
        traj_pca = pca.transform(traj_scaled)[:, :2]
        trajectories_pca.append(traj_pca)
    print("  PCA projection complete")
    
    # Project to PHATE
    trajectories_phate = []
    if PHATE_AVAILABLE and phate_op is not None:
        for traj in trajectories:
            traj_scaled = scaler.transform(traj)
            traj_phate = phate_op.transform(traj_scaled)
            trajectories_phate.append(traj_phate)
        print("  PHATE projection complete")
    
    return trajectories_pca, trajectories_phate


def plot_trajectories(embedding_2d, trajectories_list, title, figsize=(10, 8), 
                     background_colors=None, color_label=None, cmap='viridis'):
    """Plot trajectories on 2D embedding with optional property coloring."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Background points
    if background_colors is not None:
        sc = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                       c=background_colors, s=5, alpha=0.6, cmap=cmap, rasterized=True)
        cbar = plt.colorbar(sc, ax=ax, label=color_label)
    else:
        ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                   c='lightgray', s=3, alpha=0.3, rasterized=True)
    
    # Trajectories
    traj_colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories_list)))
    for i, traj in enumerate(trajectories_list):
        ax.plot(traj[:, 0], traj[:, 1], '-', color=traj_colors[i], 
               linewidth=2.5, alpha=0.9, label=f'Traj {i+1}', zorder=10)
        ax.scatter(traj[0, 0], traj[0, 1], marker='o', 
                  color=traj_colors[i], s=120, edgecolors='black', linewidths=2, zorder=15)
        ax.scatter(traj[-1, 0], traj[-1, 1], marker='*', 
                  color=traj_colors[i], s=500, edgecolors='black', linewidths=2, zorder=15)
    
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    plt.tight_layout()
    
    return fig


def visualize_results(embeddings_pca, embeddings_phate, trajectories_pca, trajectories_phate, 
                     prop_arrays=None, property_names=None, output_dir=None):
    """Visualize gradient ascent results, optionally colored by properties."""
    print("\nGenerating visualizations...")
    
    # Define property names
    if property_names is None:
        property_names = [f'Property {i+1}' for i in range(len(prop_arrays) if prop_arrays else 0)]
    
    # Visualizations without property coloring (baseline)
    if len(trajectories_pca) > 0:
        fig = plot_trajectories(
            embeddings_pca,
            trajectories_pca,
            "Gradient Ascent Paths in PCA Space\n(○ = start, ★ = end)",
            figsize=(10, 8)
        )
        if output_dir:
            pca_path = os.path.join(output_dir, 'gradient_ascent_pca.png')
            fig.savefig(pca_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {pca_path}")
        plt.show()
    
    if len(trajectories_phate) > 0 and embeddings_phate is not None:
        fig = plot_trajectories(
            embeddings_phate,
            trajectories_phate,
            "Gradient Ascent Paths in PHATE Space\n(○ = start, ★ = end)",
            figsize=(10, 8)
        )
        if output_dir:
            phate_path = os.path.join(output_dir, 'gradient_ascent_phate.png')
            fig.savefig(phate_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {phate_path}")
        plt.show()
    
    # Visualizations colored by each property
    if prop_arrays is not None and len(prop_arrays) > 0:
        print(f"\nGenerating property-colored visualizations ({len(prop_arrays)} properties)...")
        
        for prop_idx, (prop_vals, prop_name) in enumerate(zip(prop_arrays, property_names)):
            prop_vals = np.array(prop_vals)
            print(f"  Property {prop_idx + 1}/{len(prop_arrays)}: {prop_name}")
            
            # PCA colored by property
            if len(trajectories_pca) > 0:
                fig = plot_trajectories(
                    embeddings_pca,
                    trajectories_pca,
                    f"Gradient Ascent in PCA Space - Colored by {prop_name}\n(○ = start, ★ = end)",
                    figsize=(10, 8),
                    background_colors=prop_vals,
                    color_label=prop_name,
                    cmap='viridis'
                )
                if output_dir:
                    pca_prop_path = os.path.join(output_dir, f'gradient_ascent_pca_{prop_name.replace(" ", "_")}.png')
                    fig.savefig(pca_prop_path, dpi=150, bbox_inches='tight')
                    print(f"    Saved: {pca_prop_path}")
                plt.show()
            
            # PHATE colored by property
            if len(trajectories_phate) > 0 and embeddings_phate is not None:
                fig = plot_trajectories(
                    embeddings_phate,
                    trajectories_phate,
                    f"Gradient Ascent in PHATE Space - Colored by {prop_name}\n(○ = start, ★ = end)",
                    figsize=(10, 8),
                    background_colors=prop_vals,
                    color_label=prop_name,
                    cmap='viridis'
                )
                if output_dir:
                    phate_prop_path = os.path.join(output_dir, f'gradient_ascent_phate_{prop_name.replace(" ", "_")}.png')
                    fig.savefig(phate_prop_path, dpi=150, bbox_inches='tight')
                    print(f"    Saved: {phate_prop_path}")
                plt.show()


def main():
    parser = argparse.ArgumentParser(description='Gradient ascent analysis in latent space')
    parser.add_argument('--training_dir', type=str, required=True,
                        help='Path to training output directory')
    parser.add_argument('--n_trajectories', type=int, default=1,
                        help='Number of trajectories to compute')
    parser.add_argument('--n_steps', type=int, default=100,
                        help='Number of gradient ascent steps')
    parser.add_argument('--step_size', type=float, default=1e-1,
                        help='Gradient ascent step size')
    parser.add_argument('--phate_knn', type=int, default=5,
                        help='PHATE k-nearest neighbors')
    parser.add_argument('--phate_decay', type=int, default=40,
                        help='PHATE decay parameter')
    parser.add_argument('--noise_scale', type=float, default=0,
                        help='Noise scale for stochastic trajectories')
    parser.add_argument('--output_dir', type=str, default='figs/',
                        help='Output directory for saving figures')
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    embeddings, properties, scattering_coeffs, atom_percentages = load_data(args.training_dir)
    num_properties = len(properties) if properties is not None else 1
    prop_arrays = prepare_properties(properties, embeddings, num_properties=num_properties)
    
    # Load model
    input_dim = scattering_coeffs.shape[1]
    num_properties = len(prop_arrays) if len(prop_arrays) > 0 else 1
    model = load_model(args.training_dir, input_dim, num_properties, device)
    
    # Run gradient ascent
    trajectories, all_num_atoms = run_gradient_ascent(
        model, embeddings, prop_arrays, device,
        n_trajectories=args.n_trajectories,
        n_steps=args.n_steps,
        step_size=args.step_size,
        noise_scale=args.noise_scale
    )
    
    # Prepare dimensionality reduction
    print("\nPreparing dimensionality reduction...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    pca = PCA(n_components=min(50, embeddings.shape[1]))
    embeddings_pca_full = pca.fit_transform(embeddings_scaled)
    embeddings_pca = embeddings_pca_full[:, :2]
    print("  PCA complete")
    
    embeddings_phate = None
    phate_op = None
    if PHATE_AVAILABLE:
        print(f"  Computing PHATE (knn={args.phate_knn}, decay={args.phate_decay})...")
        phate_op = phate.PHATE(
            n_components=2,
            knn=args.phate_knn,
            decay=args.phate_decay,
            t='auto',
            verbose=False,
            random_state=42
        )
        embeddings_phate = phate_op.fit_transform(embeddings_scaled)
        print("  PHATE complete")
    
    # Project trajectories
    trajectories_pca, trajectories_phate = project_trajectories(
        trajectories, embeddings, scaler, pca, phate_op
    )
    
    # Define property names (last property is num_atoms if num_properties > 1)
    if num_properties > 1:
        property_names = [f'Property {i+1}' for i in range(len(prop_arrays) - 1)] + ['num_atoms']
    else:
        property_names = [f'Property {i+1}' for i in range(len(prop_arrays))]
    
    # Visualize
    visualize_results(embeddings_pca, embeddings_phate, trajectories_pca, trajectories_phate,
                      prop_arrays=prop_arrays, property_names=property_names, output_dir=args.output_dir)
    
    print("\n✓ Analysis complete!")


if __name__ == '__main__':
    main()
