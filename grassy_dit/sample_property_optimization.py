"""
Sampling script for property optimization without scaffold conditioning.
Generates molecules by optimizing in GRASSY's latent space for a target property.

Workflow:
1. Initialize latent vector (sample from prior, use mean, or reference molecule)
2. Optimize in latent space for target property (gradient ascent)
3. Decode optimized latent back to scattering moments
4. Generate molecules conditioned on optimized scattering (NO scaffold)
5. Visualize optimization trajectory in PCA space with molecules
"""
import argparse
from glob import glob
import numpy as np
import torch
import yaml
from typing import Optional, List
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, QED
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import os
import warnings
warnings.filterwarnings('ignore')

# Try to import PHATE
try:
    import phate
    PHATE_AVAILABLE = True
except ImportError:
    PHATE_AVAILABLE = False

from grassy_dit.train import ScatteringGraphDIT
from models.GRASSY_model import GRASSY
from models.LatentOptimization import LatentOptimizer
from models.ScatteringTransform import GraphScatteringTransform

from evaluation.utils import (
    compute_scattering_from_smiles,
)

def load_dit_model(checkpoint_path: str, config_path: str, device: str = "cpu", scattering_path=None):
    """Load GRASSY-DiT model from checkpoint."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = ScatteringGraphDIT(config)
    model.device = torch.device(device)

    # Load checkpoint early so we can infer dimensions if scattering file is missing
    checkpoint = torch.load(checkpoint_path, map_location=device)

    scattering_cfg = config.get('scattering', {})
    J = scattering_cfg.get('J', 4)
    num_levels = 1 + J + J * (J - 1) // 2
    num_moments = scattering_cfg.get('num_moments', 4)

    if scattering_path is not None:
        scattering_data = np.load(scattering_path)
        model.num_atom_types = scattering_data.shape[-1] // (num_levels * num_moments)
    else:
        state = checkpoint.get("model_state_dict", {})
        level_proj_w = state.get("denoiser.scatter_tokenizer.level_proj.weight")
        pos = state.get("denoiser.scatter_tokenizer.pos")

        if level_proj_w is not None:
            model.num_atom_types = level_proj_w.shape[1] // num_moments
        elif pos is not None:
            model.num_atom_types = pos.shape[1] - num_levels

    model.num_levels = num_levels
    model.num_moments = num_moments
    model.J = J

    model._initialize_model(model.model_class, checkpoint)
    model.is_fitted_ = True
    model.fitting_loss = [0.0]
    model.fitting_epoch = 0

    hparams = checkpoint.get('hyperparameters', {})
    if not getattr(model, "dataset_info", None):
        model.dataset_info = hparams.get("dataset_info", None)

    return model

def load_grassy_model(checkpoint_dir, device="cpu"):
    """Load the GRASSY autoencoder model with its config."""
    
    # Load config
    config_path = os.path.join(checkpoint_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint_path = glob(os.path.join(checkpoint_dir, 'best-epoch=*.ckpt'))[0]
    
    # Load checkpoint to extract the saved hparams
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    print("hyper_parameters:", checkpoint['hyper_parameters'])
    print("type:", type(checkpoint['hyper_parameters']))
    # Get hparams from checkpoint (Lightning saves them automatically)
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        model = GRASSY.load_from_checkpoint(checkpoint_path, hparams=hparams, map_location=device)
    else:
        raise ValueError("No hyper_parameters found in checkpoint")
    
    model.eval()
    return model, config

def get_atom_types(dit_model) -> List[str]:
    """Extract atom types from DiT model."""
    if hasattr(dit_model, "dataset_info") and dit_model.dataset_info:
        if "atom_decoder" in dit_model.dataset_info:
            return dit_model.dataset_info["atom_decoder"]
    return ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"]

def load_training_embeddings(grassy_checkpoint_dir):
    """Load training embeddings from GRASSY checkpoint directory."""
    embeddings = None
    
    if os.path.exists(grassy_checkpoint_dir):
        for f in os.listdir(grassy_checkpoint_dir):
            if f.endswith('.npy'):
                if 'ordered_embedding' in f:
                    embedding_path = os.path.join(grassy_checkpoint_dir, f)
                    embeddings = np.load(embedding_path)
                    print(f"  Loaded training embeddings: {embeddings.shape}")
                    break
    
    return embeddings

def initialize_latent(grassy_model, device, method='prior', scattering_model=None, 
                     reference_smiles=None, atom_types=None, training_embeddings=None):
    """
    Initialize latent vector for optimization.
    
    Args:
        grassy_model: GRASSY autoencoder model
        device: torch device
        method: 'prior' (sample from N(0,1)), 'mean' (use zeros), 
                'center' (use training point closest to center),
                'first_training' (use first training example),
                'reference' (encode reference molecule)
        scattering_model: Required for 'reference' method
        reference_smiles: Required for 'reference' method
        atom_types: Required for 'reference' method
        training_embeddings: Required for 'center' and 'first_training' methods
    
    Returns:
        Initial latent vector [1, latent_dim]
    """
    latent_dim = grassy_model.hparams.bottle_dim
    
    if method == 'prior':
        # Sample from standard normal (VAE prior)
        z = torch.randn(1, latent_dim, device=device)
        print(f"Initialized latent from prior N(0,1)")
    elif method == 'mean':
        # Start from mean (zeros in latent space)
        z = torch.zeros(1, latent_dim, device=device)
        print(f"Initialized latent at mean (zeros)")
    elif method == 'center':
        # Use training point closest to center of latent space
        if training_embeddings is None:
            raise ValueError("training_embeddings required for center method")
        
        embeddings_center = np.mean(training_embeddings, axis=0)
        distances_to_center = np.linalg.norm(training_embeddings - embeddings_center, axis=1)
        start_idx = np.argmin(distances_to_center)
        
        z = torch.tensor(training_embeddings[start_idx:start_idx+1], dtype=torch.float32, device=device)
        print(f"Initialized latent from training point closest to center (index {start_idx})")
    elif method == 'first_training':
        # Use first training example's latent embedding
        if training_embeddings is None:
            raise ValueError("training_embeddings required for first_training method")
        
        z = torch.tensor(training_embeddings[0:1], dtype=torch.float32, device=device)
        print(f"Initialized latent from first training example")
    elif method == 'reference':
        # Encode a reference molecule
        if reference_smiles is None or scattering_model is None or atom_types is None:
            raise ValueError("reference_smiles, scattering_model, and atom_types required for reference method")
        
        print(f"Encoding reference molecule: {reference_smiles}")
        scattering = compute_scattering_from_smiles(
            [reference_smiles], scattering_model, atom_types, device
        )
        with torch.no_grad():
            z, _, _ = grassy_model.embed(scattering)
        print(f"Reference molecule encoded to latent space")
    else:
        raise ValueError(f"Unknown initialization method: {method}")
    
    return z

def compute_molecule_properties(smiles_list):
    """
    Compute various molecular properties for evaluation.
    
    Returns:
        dict with properties for each valid molecule
    """
    properties = {
        'smiles': [],
        'valid': [],
        'qed': [],
        'sa': [],
        'logp': [],
        'mol_wt': [],
        'num_atoms': [],
    }
    
    for smi in smiles_list:
        properties['smiles'].append(smi)
        
        if smi is None:
            properties['valid'].append(False)
            properties['qed'].append(None)
            properties['sa'].append(None)
            properties['logp'].append(None)
            properties['mol_wt'].append(None)
            properties['num_atoms'].append(None)
            continue
        
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            properties['valid'].append(False)
            properties['qed'].append(None)
            properties['sa'].append(None)
            properties['logp'].append(None)
            properties['mol_wt'].append(None)
            properties['num_atoms'].append(None)
            continue
        
        properties['valid'].append(True)
        
        try:
            properties['qed'].append(QED.qed(mol))
        except:
            properties['qed'].append(None)
        
        try:
            from datasets.sascorer import calculateScore
            properties['sa'].append(calculateScore(mol))
        except:
            properties['sa'].append(None)
        
        try:
            properties['logp'].append(Descriptors.MolLogP(mol))
        except:
            properties['logp'].append(None)
        
        try:
            properties['mol_wt'].append(Descriptors.MolWt(mol))
        except:
            properties['mol_wt'].append(None)
        
        try:
            properties['num_atoms'].append(mol.GetNumAtoms())
        except:
            properties['num_atoms'].append(None)
    
    return properties

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


def visualize_optimization_trajectories(all_latent_trajectories, output_path, 
                                       training_embeddings=None, grassy_model=None, 
                                       property_idx=0, phate_knn=5, phate_decay=40):
    """
    Visualize multiple optimization trajectories in PCA and PHATE spaces.
    
    Args:
        all_latent_trajectories: List of trajectory lists (one per optimization run)
        output_path: Base path to save figures
        training_embeddings: Optional training embeddings for context
        grassy_model: GRASSY model for computing property values
        property_idx: Index of property to color by
        phate_knn: PHATE k-nearest neighbors
        phate_decay: PHATE decay parameter
    """
    print("\nGenerating trajectory visualizations...")
    
    # Convert trajectories to numpy
    trajectories_np = []
    for traj in all_latent_trajectories:
        traj_np = torch.stack(traj).squeeze(1).cpu().numpy()
        trajectories_np.append(traj_np)
    
    # Prepare embeddings for dimensionality reduction
    if training_embeddings is not None:
        print("  Using training embeddings for context")
        embeddings = training_embeddings
    else:
        print("  Using only trajectory points")
        embeddings = np.vstack(trajectories_np)
    
    # Fit scaler and PCA
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings_scaled)
    print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    
    # Project trajectories to PCA
    trajectories_pca = []
    for traj in trajectories_np:
        traj_scaled = scaler.transform(traj)
        traj_pca = pca.transform(traj_scaled)
        trajectories_pca.append(traj_pca)
    
    # Compute property values for background coloring
    property_values = None
    if grassy_model is not None and training_embeddings is not None:
        print(f"  Computing property values for background coloring...")
        with torch.no_grad():
            embeddings_torch = torch.tensor(training_embeddings, dtype=torch.float32, device=grassy_model.device)
            y_full, _, _ = grassy_model.predict(embeddings_torch)
            property_values = y_full[:, property_idx].cpu().numpy()
    
    # Plot PCA without property coloring
    fig = plot_trajectories(
        embeddings_pca,
        trajectories_pca,
        f"Gradient Ascent Paths in PCA Space\n(○ = start, ★ = end)",
        figsize=(10, 8)
    )
    pca_path = output_path.replace('.png', '_pca.png')
    fig.savefig(pca_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {pca_path}")
    plt.close()
    
    # Plot PCA with property coloring
    if property_values is not None:
        fig = plot_trajectories(
            embeddings_pca,
            trajectories_pca,
            f"Gradient Ascent in PCA Space - Colored by Property {property_idx}\n(○ = start, ★ = end)",
            figsize=(10, 8),
            background_colors=property_values,
            color_label=f'Property {property_idx}',
            cmap='viridis'
        )
        pca_prop_path = output_path.replace('.png', f'_pca_property{property_idx}.png')
        fig.savefig(pca_prop_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {pca_prop_path}")
        plt.close()
    
    # PHATE visualization if available
    if PHATE_AVAILABLE and training_embeddings is not None:
        print(f"  Computing PHATE (knn={phate_knn}, decay={phate_decay})...")
        phate_op = phate.PHATE(
            n_components=2,
            knn=phate_knn,
            decay=phate_decay,
            t='auto',
            verbose=False,
            random_state=42
        )
        embeddings_phate = phate_op.fit_transform(embeddings_scaled)
        
        # Project trajectories to PHATE
        trajectories_phate = []
        for traj in trajectories_np:
            traj_scaled = scaler.transform(traj)
            traj_phate = phate_op.transform(traj_scaled)
            trajectories_phate.append(traj_phate)
        
        # Plot PHATE without property coloring
        fig = plot_trajectories(
            embeddings_phate,
            trajectories_phate,
            f"Gradient Ascent Paths in PHATE Space\n(○ = start, ★ = end)",
            figsize=(10, 8)
        )
        phate_path = output_path.replace('.png', '_phate.png')
        fig.savefig(phate_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {phate_path}")
        plt.close()
        
        # Plot PHATE with property coloring
        if property_values is not None:
            fig = plot_trajectories(
                embeddings_phate,
                trajectories_phate,
                f"Gradient Ascent in PHATE Space - Colored by Property {property_idx}\n(○ = start, ★ = end)",
                figsize=(10, 8),
                background_colors=property_values,
                color_label=f'Property {property_idx}',
                cmap='viridis'
            )
            phate_prop_path = output_path.replace('.png', f'_phate_property{property_idx}.png')
            fig.savefig(phate_prop_path, dpi=150, bbox_inches='tight')
            print(f"  Saved: {phate_prop_path}")
            plt.close()
    
    print("  Trajectory visualizations complete!")

def print_metrics_table(properties, property_idx, initial_property=None):
    """
    Print a table of metrics for paper.
    
    Args:
        properties: Dict of computed properties
        property_idx: Index of optimized property
        initial_property: Starting property value (if available)
    """
    valid_count = sum(properties['valid'])
    total_count = len(properties['valid'])
    
    print("\n" + "="*60)
    print("EVALUATION METRICS FOR PAPER")
    print("="*60)
    
    # Basic statistics
    print(f"\nGeneration Statistics:")
    print(f"  Total Generated:     {total_count}")
    print(f"  Valid:               {valid_count} ({100*valid_count/total_count:.1f}%)")
    print(f"  Invalid:             {total_count - valid_count}")
    
    if valid_count == 0:
        print("\nNo valid molecules generated!")
        return
    
    # Uniqueness
    unique_smiles = set([s for s in properties['smiles'] if s is not None])
    print(f"  Unique:              {len(unique_smiles)} ({100*len(unique_smiles)/valid_count:.1f}%)")
    
    # Property statistics
    property_names = ['QED', 'SA Score', 'LogP', 'Mol Weight', 'Num Atoms']
    property_keys = ['qed', 'sa', 'logp', 'mol_wt', 'num_atoms']
    
    print(f"\nMolecular Properties (Mean ± Std):")
    for name, key in zip(property_names, property_keys):
        values = [v for v in properties[key] if v is not None]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {name:15s}  {mean_val:.3f} ± {std_val:.3f}")
    
    # Optimization success
    if property_idx == 0:  # QED
        opt_values = [v for v in properties['qed'] if v is not None]
        if opt_values:
            print(f"\nTarget Property Optimization (QED):")
            print(f"  Mean:                {np.mean(opt_values):.3f}")
            print(f"  Max:                 {np.max(opt_values):.3f}")
            print(f"  Min:                 {np.min(opt_values):.3f}")
            if initial_property is not None:
                improvement = np.mean(opt_values) - initial_property
                print(f"  Initial:             {initial_property:.3f}")
                print(f"  Improvement:         {improvement:+.3f} ({100*improvement/abs(initial_property):.1f}%)")
    
    print("\n" + "="*60)
    print("\nMetrics for LaTeX table:")
    print(f"Validity & {100*valid_count/total_count:.1f}\\%% \\\\")
    print(f"Uniqueness & {100*len(unique_smiles)/valid_count:.1f}\\%% \\\\")
    for name, key in zip(property_names, property_keys):
        values = [v for v in properties[key] if v is not None]
        if values:
            print(f"{name} & ${np.mean(values):.3f} \\pm {np.std(values):.3f}$ \\\\")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Generate molecules via property optimization')
    parser.add_argument('--checkpoint', required=True, help='Path to DiT model checkpoint')
    parser.add_argument('--config', default='configs/ZINC/BBAB/BBAB_dit_config.yaml', help='Path to config yaml')
    parser.add_argument('--grassy_checkpoint_dir', required=True, help='Path to GRASSY autoencoder checkpoint')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples per scattering')
    parser.add_argument('--num_nodes', type=int, default=None, help='Number of atoms (None = predict from scattering)')
    parser.add_argument('--output', default='generated_property_opt.txt', help='Output file')
    parser.add_argument('--save_trajectory', action='store_true', help='Save optimization trajectory')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--property_idx', type=int, default=0, help='Index of property to optimize')
    parser.add_argument('--step_size_latent', type=float, default=1e-1, help='Step size for latent optimization')
    parser.add_argument('--n_steps_latent', type=int, default=40, help='Number of gradient ascent steps')
    parser.add_argument('--n_trajectories', type=int, default=1, help='Number of optimization trajectories to compute')
    parser.add_argument('--noise_scale', type=float, default=0.0, help='Noise scale for stochastic gradient ascent')
    parser.add_argument('--phate_knn', type=int, default=5, help='PHATE k-nearest neighbors')
    parser.add_argument('--phate_decay', type=int, default=40, help='PHATE decay parameter')
    parser.add_argument('--init_method', default='center', 
                        choices=['prior', 'mean', 'center', 'first_training', 'reference'], 
                        help='Latent initialization method')
    parser.add_argument('--reference_smiles', default=None, 
                        help='Reference molecule SMILES (for init_method=reference)')
    parser.add_argument('--visualize', default=True, action='store_true', help='Generate PCA visualization')
    args = parser.parse_args()
    
    device = args.device
    print(f"Using device: {device}")
    
    # Load DiT model
    print("\nLoading GRASSY-DiT model...")
    dit_model = load_dit_model(args.checkpoint, args.config, device)

    # Load GRASSY autoencoder
    print("Loading GRASSY model...")
    grassy_model, grassy_configs = load_grassy_model(args.grassy_checkpoint_dir, device)

    # Load scattering model for reference molecule encoding
    atom_types = get_atom_types(dit_model)
    scattering_model = None
    if args.init_method == 'reference':
        from models.ScatteringTransform import GraphScatteringTransform
        scattering_model = GraphScatteringTransform(
            in_channels=len(atom_types),
            J=dit_model.J,
            num_moments=dit_model.num_moments
        ).to(device)
        scattering_model.eval()

    # Load training embeddings for center/first_training initialization
    training_embeddings = None
    if args.init_method in ['center', 'first_training'] or args.visualize:
        print("\nLoading training embeddings...")
        training_embeddings = load_training_embeddings(args.grassy_checkpoint_dir)
        if training_embeddings is None and args.init_method in ['center', 'first_training']:
            raise ValueError(f"Could not load training embeddings from {args.grassy_checkpoint_dir}")

    # Initialize latent vector
    print(f"\nInitializing latent vector with method: {args.init_method}")
    z = initialize_latent(
        grassy_model, 
        device, 
        method=args.init_method,
        scattering_model=scattering_model,
        reference_smiles=args.reference_smiles,
        atom_types=atom_types,
        training_embeddings=training_embeddings
    )
    print(f"Latent dimension: {z.shape}")

    # Compute initial property value for comparison
    initial_property = None
    with torch.no_grad():
        y_full, _, _ = grassy_model.predict(z)
        initial_property = y_full[0, args.property_idx].item()
        print(f"Initial property value: {initial_property:.4f}")

    # Run multiple optimization trajectories
    print(f"\nRunning {args.n_trajectories} optimization trajectories...")
    print(f"  Property: {args.property_idx}")
    print(f"  Steps per trajectory: {args.n_steps_latent}")
    print(f"  Step size: {args.step_size_latent}")
    print(f"  Noise scale: {args.noise_scale}")
    
    all_latent_trajectories = []
    all_decoded_trajectories = []
    all_num_atoms_trajectories = []
    all_final_properties = []
    all_smiles = []
    all_properties = []
    
    for traj_idx in range(args.n_trajectories):
        print(f"\n--- Trajectory {traj_idx + 1}/{args.n_trajectories} ---")
        
        # Set up optimizer with noise for this trajectory
        latent_optimizer = LatentOptimizer(
            grassy_model,
            property_idx=args.property_idx,
            step_size=args.step_size_latent,
            noise_scale=args.noise_scale
        )
        
        # Optimization by gradient ascent in latent space
        latent_trajectory, decoded_trajectory, num_atoms_trajectory = latent_optimizer.optimize(
            z, n_steps=args.n_steps_latent, return_decoded=True
        )
        
        # Store trajectories
        all_latent_trajectories.append(latent_trajectory)
        all_decoded_trajectories.append(decoded_trajectory)
        all_num_atoms_trajectories.append(num_atoms_trajectory)
        
        # Compute final property value
        with torch.no_grad():
            y_full, _, _ = grassy_model.predict(latent_trajectory[-1])
            final_property = y_full[0, args.property_idx].item()
            all_final_properties.append(final_property)
            print(f"  Final property: {final_property:.4f} (Δ = {final_property - initial_property:+.4f})")
        
        # Get final optimized scattering
        scattering_final = decoded_trajectory[-1]
        num_atoms_pred = num_atoms_trajectory[-1]
        
        # Determine number of atoms to generate
        if args.num_nodes is None:
            num_nodes = int(round(num_atoms_pred.item()))
            print(f"  Predicted num_atoms: {num_atoms_pred.item():.1f} → {num_nodes}")
        else:
            num_nodes = args.num_nodes
            print(f"  Using specified num_nodes: {num_nodes}")
        
        # Convert scattering to numpy for generation
        scattering_np = scattering_final.detach().cpu().numpy()
        if scattering_np.ndim == 2:
            scattering_np = scattering_np[0]  # Remove batch dimension
        
        # Generate molecules WITHOUT scaffold conditioning
        print(f"  Generating {args.num_samples} molecules...")
        smiles_list = dit_model.generate(
            scattering=scattering_np,
            num_nodes=num_nodes,
            batch_size=args.num_samples,
            scaffold_X=None,
            scaffold_E=None,
            scaffold_node_mask=None,
        )
        
        # Compute properties for evaluation
        properties = compute_molecule_properties(smiles_list)
        valid_count = sum(properties['valid'])
        print(f"  Generated {valid_count}/{len(smiles_list)} valid molecules")
        
        all_smiles.extend(smiles_list)
        all_properties.append(properties)
    
    print(f"\n✓ All {args.n_trajectories} trajectories complete!")
    
    # Aggregate results
    avg_final_property = np.mean(all_final_properties)
    std_final_property = np.std(all_final_properties)
    print(f"\nFinal property value: {avg_final_property:.4f} ± {std_final_property:.4f}")
    print(f"Property improvement: {avg_final_property - initial_property:+.4f}")
    print(f"Total molecules generated: {len(all_smiles)}")
    
    # Save trajectory if requested
    if args.save_trajectory:
        trajectory_file = args.output.replace('.txt', '_trajectory.npz')
        np.savez(
            trajectory_file,
            latent=[[lat.cpu().numpy() for lat in traj] for traj in all_latent_trajectories],
            scattering=[[scat.cpu().numpy() for scat in traj] for traj in all_decoded_trajectories],
            num_atoms=[[na.cpu().numpy() for na in traj] for traj in all_num_atoms_trajectories],
            final_properties=all_final_properties
        )
        print(f"Trajectories saved to {trajectory_file}")
    
    # Save results
    valid_smiles = [s for s in all_smiles if s is not None]
    with open(args.output, 'w') as f:
        f.write(f"# Property optimized: {args.property_idx}\n")
        f.write(f"# Number of trajectories: {args.n_trajectories}\n")
        f.write(f"# Optimization steps per trajectory: {args.n_steps_latent}\n")
        f.write(f"# Noise scale: {args.noise_scale}\n")
        f.write(f"# Initialization method: {args.init_method}\n")
        f.write(f"# Initial property: {initial_property:.4f}\n")
        f.write(f"# Final property (mean): {avg_final_property:.4f}\n")
        f.write(f"# Final property (std): {std_final_property:.4f}\n")
        f.write(f"# Improvement: {avg_final_property - initial_property:+.4f}\n")
        f.write(f"# Total molecules: {len(valid_smiles)}\n")
        f.write("#\n")
        for smi in valid_smiles:
            f.write(smi + '\n')
    
    print(f"Saved {len(valid_smiles)} molecules to {args.output}")
    
    # Aggregate properties for metrics table
    aggregated_properties = {
        'smiles': all_smiles,
        'valid': [],
        'qed': [],
        'sa': [],
        'logp': [],
        'mol_wt': [],
        'num_atoms': [],
    }
    for prop_dict in all_properties:
        for key in ['valid', 'qed', 'sa', 'logp', 'mol_wt', 'num_atoms']:
            aggregated_properties[key].extend(prop_dict[key])
    
    # Print metrics table
    print_metrics_table(aggregated_properties, args.property_idx, initial_property)
    
    # Generate visualization
    if args.visualize:
        viz_path = args.output.replace('.txt', '_trajectory.png')
        print(f"\nGenerating trajectory visualizations...")
        visualize_optimization_trajectories(
            all_latent_trajectories,
            viz_path,
            training_embeddings=training_embeddings,
            grassy_model=grassy_model,
            property_idx=args.property_idx,
            phate_knn=args.phate_knn,
            phate_decay=args.phate_decay
        )

    # Print some generated molecules
    if valid_smiles:
        print("\nSample generated molecules:")
        for i, smi in enumerate(valid_smiles[:5]):
            print(f"  {i+1}. {smi}")


if __name__ == "__main__":
    main()

# Example usage:
#
# Basic: Single trajectory, start from center of latent space
# python -m grassy_dit.sample_property_optimization \
#     --checkpoint checkpoints_dit/checkpoints_zinc_bbab_checkpoint_best.pt \
#     --grassy_checkpoint_dir outputs/BBAB_fixed_2026-01-27-16-11-28/ \
#     --property_idx 0 \
#     --n_steps_latent 50 \
#     --num_samples 10 \
#     --visualize \
#     --output generated_prop0_opt.txt
#
# Multiple noisy trajectories:
# python -m grassy_dit.sample_property_optimization \
#     --checkpoint checkpoints_dit/checkpoints_zinc_bbab_checkpoint_best.pt \
#     --grassy_checkpoint_dir outputs/BBAB_fixed_2026-01-27-16-11-28/ \
#     --property_idx 0 \
#     --n_trajectories 5 \
#     --n_steps_latent 50 \
#     --noise_scale 0.01 \
#     --num_samples 10 \
#     --visualize
#
# Start from reference molecule:
# python -m grassy_dit.sample_property_optimization \
#     --checkpoint checkpoints_dit/checkpoints_zinc_bbab_checkpoint_best.pt \
#     --grassy_checkpoint_dir outputs/BBAB_fixed_2026-01-27-16-11-28/ \
#     --property_idx 0 \
#     --init_method reference \
#     --reference_smiles "c1ccccc1" \
#     --n_trajectories 3 \
#     --n_steps_latent 50 \
#     --noise_scale 0.01 \
#     --visualize
#
# With PHATE visualization and trajectory saving:
# python -m grassy_dit.sample_property_optimization \
#     --checkpoint checkpoints_dit/checkpoints_zinc_bbab_checkpoint_best.pt \
#     --grassy_checkpoint_dir outputs/BBAB_fixed_2026-01-27-16-11-28/ \
#     --property_idx 0 \
#     --n_trajectories 5 \
#     --noise_scale 0.01 \
#     --phate_knn 5 \
#     --phate_decay 40 \
#     --save_trajectory \
#     --visualize
