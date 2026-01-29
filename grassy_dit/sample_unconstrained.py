"""
Sampling script for molecule generation via random sampling from latent space.
Generates molecules by sampling from the prior distribution in GRASSY's latent space.

Computes metrics for unconditional generation evaluation:
- Validity (Val): % of generated SMILES that are valid molecules
- Uniqueness (Uniq): % of valid molecules that are unique
- Novelty (Nov): % of unique molecules not in training set
- Diversity (Div): Average pairwise Tanimoto distance among generated molecules
- Similarity (Sim): Average max Tanimoto similarity to training set
- FCD: Fréchet ChemNet Distance to training set

Workflow:
1. Sample latent vectors from prior N(0, I)
2. Decode latent vectors to scattering moments
3. Generate molecules conditioned on scattering (NO scaffold)
4. Evaluate generated molecules with all metrics
5. Print results in table format
"""
import argparse
from glob import glob
import numpy as np
import torch
import yaml
from typing import List, Set, Optional
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter

import os
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Try to import FCD
try:
    from fcd import get_fcd, load_ref_model, canonical_smiles
    FCD_AVAILABLE = True
    USE_FCD_TORCH = False
except ImportError:
    try:
        from fcd_torch import FCD
        FCD_AVAILABLE = True
        USE_FCD_TORCH = True
    except ImportError:
        FCD_AVAILABLE = False
        USE_FCD_TORCH = False

from grassy_dit.train import ScatteringGraphDIT
from models.GRASSY_model import GRASSY


def load_dit_model(checkpoint_path: str, config_path: str, device: str = "cpu", scattering_path=None):
    """Load GRASSY-DiT model from checkpoint."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = ScatteringGraphDIT(config)
    model.device = torch.device(device)

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
    
    config_path = os.path.join(checkpoint_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint_path = glob(os.path.join(checkpoint_dir, 'best-epoch=*.ckpt'))[0]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    print("hyper_parameters:", checkpoint['hyper_parameters'])
    
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


def load_training_smiles(training_smiles_path: str) -> Set[str]:
    """Load training SMILES for novelty computation."""
    training_smiles = set()
    
    if training_smiles_path and os.path.exists(training_smiles_path):
        with open(training_smiles_path, 'r') as f:
            for line in f:
                smi = line.strip()
                if smi and not smi.startswith('#'):
                    # Canonicalize
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        can_smi = Chem.MolToSmiles(mol)
                        training_smiles.add(can_smi)
        print(f"  Loaded {len(training_smiles)} training SMILES")
    
    return training_smiles


def sample_from_prior(grassy_model, n_samples: int, device: str, 
                      training_embeddings=None, method='prior', noise_scale=0.1):
    """
    Sample latent vectors from various distributions.
    
    Args:
        method: 'prior' (N(0,I)), 'training' (resample training points),
                'noisy_training' (training + noise), 'kde' (kernel density estimate)
        noise_scale: Std of noise to add for 'noisy_training' method
    """
    latent_dim = grassy_model.hparams.bottle_dim
    
    if method == 'prior':
        # Original: sample from standard normal
        z = torch.randn(n_samples, latent_dim, device=device)
        print(f"Sampled {n_samples} latent vectors from prior N(0,1)")
        
    elif method == 'training':
        # Resample from training embeddings (bootstrap)
        if training_embeddings is None:
            raise ValueError("training_embeddings required for 'training' method")
        indices = np.random.choice(len(training_embeddings), n_samples, replace=True)
        z = torch.tensor(training_embeddings[indices], dtype=torch.float32, device=device)
        print(f"Sampled {n_samples} latent vectors from training distribution")
        
    elif method == 'noisy_training':
        # Training points + small Gaussian noise
        if training_embeddings is None:
            raise ValueError("training_embeddings required for 'noisy_training' method")
        indices = np.random.choice(len(training_embeddings), n_samples, replace=True)
        z_np = training_embeddings[indices] + noise_scale * np.random.randn(n_samples, latent_dim)
        z = torch.tensor(z_np, dtype=torch.float32, device=device)
        print(f"Sampled {n_samples} latent vectors from noisy training (σ={noise_scale})")
        
    elif method == 'kde':
        # Kernel Density Estimation sampling (stays in dense regions)
        if training_embeddings is None:
            raise ValueError("training_embeddings required for 'kde' method")
        from scipy.stats import gaussian_kde
        
        # Fit KDE on training embeddings (transpose for scipy)
        kde = gaussian_kde(training_embeddings.T, bw_method='scott')
        z_np = kde.resample(n_samples).T  # Shape: [n_samples, latent_dim]
        z = torch.tensor(z_np, dtype=torch.float32, device=device)
        print(f"Sampled {n_samples} latent vectors from KDE")
        
    elif method == 'dense_regions':
        # Sample preferentially from high-density regions using nearest neighbor density
        if training_embeddings is None:
            raise ValueError("training_embeddings required for 'dense_regions' method")
        from sklearn.neighbors import NearestNeighbors
        
        # Estimate density via k-NN distances
        k = min(10, len(training_embeddings) - 1)
        nn = NearestNeighbors(n_neighbors=k+1).fit(training_embeddings)
        distances, _ = nn.kneighbors(training_embeddings)
        avg_distances = distances[:, 1:].mean(axis=1)  # Exclude self
        
        # Convert distances to density weights (smaller distance = higher density)
        density_weights = 1.0 / (avg_distances + 1e-6)
        density_weights /= density_weights.sum()
        
        # Sample indices weighted by density
        indices = np.random.choice(len(training_embeddings), n_samples, 
                                   replace=True, p=density_weights)
        z = torch.tensor(training_embeddings[indices], dtype=torch.float32, device=device)
        print(f"Sampled {n_samples} latent vectors from dense regions")
        
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    
    return z


def decode_latent_to_scattering(grassy_model, z):
    """
    Decode latent vectors to scattering moments using GRASSY.
    
    Args:
        grassy_model: GRASSY autoencoder
        z: Latent vectors [batch, latent_dim]
    
    Returns:
        scattering: Decoded scattering moments
        num_atoms: Predicted number of atoms
    """
    with torch.no_grad():
        y_full, scattering, num_atoms = grassy_model.predict(z)
        scattering = grassy_model.decode(z)
        num_atoms = y_full[:, -1] + 7  # Last property is num_atoms an we sum for each dataset [BBAB -> 7,FBAB -> 16, JBCD -> 28]
        

    return scattering, num_atoms, y_full


def get_fingerprint(mol, radius=2, n_bits=2048):
    """Compute Morgan fingerprint for a molecule."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def compute_tanimoto_similarity(fp1, fp2):
    """Compute Tanimoto similarity between two fingerprints."""
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def compute_metrics(generated_smiles: List[str], 
                   training_smiles: Optional[Set[str]] = None,
                   training_smiles_list: Optional[List[str]] = None) -> dict:
    """
    Compute all generation quality metrics.
    
    Args:
        generated_smiles: List of generated SMILES (may contain None for failures)
        training_smiles: Set of training SMILES for novelty computation
        training_smiles_list: List of training SMILES for FCD and similarity
    
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Filter out None values
    attempted = [s for s in generated_smiles if s is not None]
    total_generated = len(generated_smiles)
    
    # Validity: % of generated that are valid
    valid_smiles = []
    valid_mols = []
    for smi in attempted:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            can_smi = Chem.MolToSmiles(mol)
            valid_smiles.append(can_smi)
            valid_mols.append(mol)
    
    validity = 100.0 * len(valid_smiles) / total_generated if total_generated > 0 else 0.0
    metrics['validity'] = validity
    metrics['n_valid'] = len(valid_smiles)
    metrics['n_total'] = total_generated
    
    if len(valid_smiles) == 0:
        metrics['uniqueness'] = 0.0
        metrics['novelty'] = 0.0
        metrics['diversity'] = 0.0
        metrics['similarity'] = 0.0
        metrics['fcd'] = float('inf')
        return metrics
    
    # Uniqueness: % of valid that are unique
    unique_smiles = list(set(valid_smiles))
    uniqueness = 100.0 * len(unique_smiles) / len(valid_smiles)
    metrics['uniqueness'] = uniqueness
    metrics['n_unique'] = len(unique_smiles)
    
    # Novelty: % of unique not in training set
    if training_smiles is not None and len(training_smiles) > 0:
        novel_smiles = [s for s in unique_smiles if s not in training_smiles]
        novelty = 100.0 * len(novel_smiles) / len(unique_smiles) if len(unique_smiles) > 0 else 0.0
        metrics['n_novel'] = len(novel_smiles)
    else:
        novelty = 100.0  # If no training set, assume all are novel
        metrics['n_novel'] = len(unique_smiles)
    metrics['novelty'] = novelty
    
    # Compute fingerprints for unique molecules
    unique_mols = [Chem.MolFromSmiles(s) for s in unique_smiles]
    unique_fps = [get_fingerprint(mol) for mol in unique_mols if mol is not None]
    
    # Diversity: Average pairwise Tanimoto distance (1 - similarity)
    if len(unique_fps) > 1:
        pairwise_distances = []
        for i in range(len(unique_fps)):
            for j in range(i + 1, len(unique_fps)):
                sim = compute_tanimoto_similarity(unique_fps[i], unique_fps[j])
                pairwise_distances.append(1.0 - sim)
        diversity = 100.0 * np.mean(pairwise_distances)
    else:
        diversity = 0.0
    metrics['diversity'] = diversity
    
    # Similarity: Average maximum Tanimoto similarity to training set
    if training_smiles_list is not None and len(training_smiles_list) > 0:
        print("  Computing similarity to training set...")
        # Sample training molecules if too many
        max_train_samples = 5000
        if len(training_smiles_list) > max_train_samples:
            train_sample = np.random.choice(training_smiles_list, max_train_samples, replace=False)
        else:
            train_sample = training_smiles_list
        
        train_fps = []
        for smi in train_sample:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                train_fps.append(get_fingerprint(mol))
        
        if len(train_fps) > 0:
            max_similarities = []
            for gen_fp in unique_fps:
                max_sim = max(compute_tanimoto_similarity(gen_fp, train_fp) for train_fp in train_fps)
                max_similarities.append(max_sim)
            similarity = 100.0 * np.mean(max_similarities)
        else:
            similarity = 0.0
    else:
        similarity = 0.0
    metrics['similarity'] = similarity
    
    # FCD: Fréchet ChemNet Distance
    if FCD_AVAILABLE and training_smiles_list is not None and len(training_smiles_list) > 0:
        print("  Computing FCD...")
        try:
            if USE_FCD_TORCH:
                fcd_calculator = FCD(device='cuda' if torch.cuda.is_available() else 'cpu')
                fcd_value = fcd_calculator(training_smiles_list, unique_smiles)
            else:
                fcd_value = get_fcd(training_smiles_list, unique_smiles)
            metrics['fcd'] = fcd_value
        except Exception as e:
            print(f"  Warning: FCD computation failed: {e}")
            metrics['fcd'] = float('nan')
    else:
        metrics['fcd'] = float('nan')
    
    return metrics


def print_metrics_table(metrics: dict, dataset_name: str = "BBAB"):
    """
    Print metrics in a nice table format matching the paper.
    
    Args:
        metrics: Dictionary with computed metrics
        dataset_name: Name of the dataset (BBAB, FBAB, JBCD)
    """
    print("\n" + "="*80)
    print("UNCONDITIONAL GENERATION METRICS")
    print("="*80)
    
    # Basic statistics
    print(f"\nGeneration Statistics:")
    print(f"  Total Generated:     {metrics['n_total']}")
    print(f"  Valid:               {metrics['n_valid']}")
    print(f"  Unique:              {metrics.get('n_unique', 'N/A')}")
    print(f"  Novel:               {metrics.get('n_novel', 'N/A')}")
    
    # Main metrics table
    print("\n" + "-"*80)
    print(f"{'Metric':<15} {'Value':>15} {'Description':<45}")
    print("-"*80)
    print(f"{'Validity':<15} {metrics['validity']:>14.2f}% {'% of generated that are valid molecules':<45}")
    print(f"{'Uniqueness':<15} {metrics['uniqueness']:>14.2f}% {'% of valid that are unique':<45}")
    print(f"{'Novelty':<15} {metrics['novelty']:>14.2f}% {'% of unique not in training set':<45}")
    print(f"{'Diversity':<15} {metrics['diversity']:>14.2f}% {'Avg pairwise Tanimoto distance × 100':<45}")
    print(f"{'Similarity':<15} {metrics['similarity']:>14.2f}% {'Avg max Tanimoto sim to training × 100':<45}")
    
    fcd_str = f"{metrics['fcd']:.4f}" if not np.isnan(metrics['fcd']) else "N/A"
    print(f"{'FCD':<15} {fcd_str:>15} {'Fréchet ChemNet Distance (lower is better)':<45}")
    print("-"*80)
    
    # LaTeX table row
    print("\n" + "="*80)
    print("LATEX TABLE ROW (for paper)")
    print("="*80)
    
    fcd_latex = f"{metrics['fcd']:.2f}" if not np.isnan(metrics['fcd']) else "N/A"
    
    print(f"""
% Row for {dataset_name} dataset:
& \\textbf{{SCULPT}} & \\textbf{{{metrics['validity']:.2f}}} & \\textbf{{{metrics['uniqueness']:.2f}}} & \\textbf{{{metrics['novelty']:.2f}}} & \\textbf{{{metrics['diversity']:.2f}}} & \\textbf{{{metrics['similarity']:.2f}}} & \\textbf{{{fcd_latex}}} \\\\
""")
    
    # Also print a cleaner version
    print("\nClean format:")
    print(f"SCULPT | Val: {metrics['validity']:.2f}% | Uniq: {metrics['uniqueness']:.2f}% | "
          f"Nov: {metrics['novelty']:.2f}% | Div: {metrics['diversity']:.2f}% | "
          f"Sim: {metrics['similarity']:.2f}% | FCD: {fcd_latex}")
    
    print("="*80 + "\n")
    
    return metrics


def visualize_samples(sampled_latents, output_path, training_embeddings=None):
    """
    Visualize sampled latent vectors in PCA space.
    """
    print("\nGenerating PCA visualization...")
    
    sampled_np = sampled_latents.cpu().numpy()
    
    if training_embeddings is not None:
        combined = np.vstack([training_embeddings, sampled_np])
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined)
        
        pca = PCA(n_components=2)
        combined_pca = pca.fit_transform(combined_scaled)
        
        training_pca = combined_pca[:len(training_embeddings)]
        sampled_pca = combined_pca[len(training_embeddings):]
    else:
        scaler = StandardScaler()
        sampled_scaled = scaler.fit_transform(sampled_np)
        pca = PCA(n_components=2)
        sampled_pca = pca.fit_transform(sampled_scaled)
        training_pca = None
    
    print(f"  PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if training_pca is not None:
        ax.scatter(training_pca[:, 0], training_pca[:, 1], 
                  c='lightgray', s=5, alpha=0.3, label='Training data', rasterized=True)
    
    ax.scatter(sampled_pca[:, 0], sampled_pca[:, 1], 
              c='blue', s=50, alpha=0.7, edgecolors='black', 
              linewidths=1, label='Prior samples', zorder=10)
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('Prior Samples in PCA Space')
    ax.legend(loc='best')
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate molecules via prior sampling and compute metrics')
    parser.add_argument('--checkpoint', required=True, help='Path to DiT model checkpoint')
    parser.add_argument('--config', default='configs/ZINC/BBAB/BBAB_dit_config.yaml', help='Path to config yaml')
    parser.add_argument('--grassy_checkpoint_dir', required=True, help='Path to GRASSY autoencoder checkpoint')
    parser.add_argument('--training_smiles', default=None, help='Path to training SMILES file (for novelty/similarity/FCD)')
    parser.add_argument('--dataset_name', default='BBAB', choices=['BBAB', 'FBAB', 'JBCD'], help='Dataset name for table')
    parser.add_argument('--n_latent_samples', type=int, default=1000, help='Number of latent samples from prior')
    parser.add_argument('--num_samples_per_latent', type=int, default=1, help='Number of molecules per latent sample')
    parser.add_argument('--num_nodes', type=int, default=None, help='Number of atoms (None = predict from scattering)')
    parser.add_argument('--output', default='generated_prior_samples.txt', help='Output file')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--visualize', action='store_true', help='Generate PCA visualization')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--sampling_method', default='prior',
                        choices=['prior', 'training', 'noisy_training', 'kde', 'dense_regions'],
                        help='Latent sampling method')
    parser.add_argument('--noise_scale', type=float, default=0.1,
                        help='Noise scale for noisy_training method')

    args = parser.parse_args()
    
    device = args.device
    print(f"Using device: {device}")
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")
    
    # Load DiT model
    print("\nLoading GRASSY-DiT model...")
    dit_model = load_dit_model(args.checkpoint, args.config, device)

    # Load GRASSY autoencoder
    print("Loading GRASSY model...")
    grassy_model, grassy_configs = load_grassy_model(args.grassy_checkpoint_dir, device)

    # Load training data for metrics
    training_smiles_set = None
    training_smiles_list = None
    if args.training_smiles:
        print("\nLoading training SMILES...")
        training_smiles_set = load_training_smiles(args.training_smiles)
        # Also keep as list for FCD
        with open(args.training_smiles, 'r') as f:
            training_smiles_list = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    # Load training embeddings for visualization
    training_embeddings = None
    if args.visualize:
        print("\nLoading training embeddings for visualization...")
        training_embeddings = load_training_embeddings(args.grassy_checkpoint_dir)

    # Sample from prior
    print(f"\nSampling {args.n_latent_samples} latent vectors using '{args.sampling_method}' method...")
    z = sample_from_prior(
        grassy_model, 
        args.n_latent_samples, 
        device,
        training_embeddings=training_embeddings,
        method=args.sampling_method,
        noise_scale=args.noise_scale
    )
    print(f"Latent shape: {z.shape}")

    # Decode to scattering
    print("\nDecoding latent vectors to scattering moments...")
    scattering, num_atoms, properties = decode_latent_to_scattering(grassy_model, z)
    print(f"Scattering shape: {scattering.shape}")
    if num_atoms is not None:
        print(f"Predicted num_atoms range: [{num_atoms.min().item():.1f}, {num_atoms.max().item():.1f}]")

    # Generate molecules using DiT
    print(f"\nGenerating molecules with DiT...")
    all_smiles = []
    
    print(f"\nGenerating molecules with DiT (batched)...")

    # Prepare all scattering vectors and num_atoms
    all_scatterings = []
    all_num_atoms = []

    for i in range(args.n_latent_samples):
        scat_i = scattering[i].cpu().numpy()
        
        if args.num_nodes is not None:
            n_atoms = args.num_nodes
        else:
            n_atoms = int(num_atoms[i].item()) if num_atoms is not None else 20
            n_atoms = max(5, min(n_atoms, 50))
        
        all_scatterings.append(scat_i)
        all_num_atoms.append(n_atoms)

    # Generate in larger batches
    all_smiles = []
    batch_size = args.batch_size  # e.g., 32 or 64

    for batch_start in range(0, args.n_latent_samples, batch_size):
        batch_end = min(batch_start + batch_size, args.n_latent_samples)
        print(f"  Processing batch {batch_start}-{batch_end}...")
        
        for i in range(batch_start, batch_end):
            try:
                smiles_list = dit_model.generate(
                    scattering=all_scatterings[i],
                    num_nodes=all_num_atoms[i],
                    batch_size=args.num_samples_per_latent,
                )
                all_smiles.extend(smiles_list)
            except Exception as e:
                all_smiles.extend([None] * args.num_samples_per_latent)
    
    print(f"\nGeneration complete. Total attempts: {len(all_smiles)}")
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(all_smiles, training_smiles_set, training_smiles_list)
    
    # Print metrics table
    print_metrics_table(metrics, args.dataset_name)
    
    # Save results
    valid_smiles = [s for s in all_smiles if s is not None and Chem.MolFromSmiles(s) is not None]
    with open(args.output, 'w') as f:
        f.write(f"# SCULPT Prior Sampling Results\n")
        f.write(f"# Dataset: {args.dataset_name}\n")
        f.write(f"# Number of latent samples: {args.n_latent_samples}\n")
        f.write(f"# Samples per latent: {args.num_samples_per_latent}\n")
        f.write(f"# Validity: {metrics['validity']:.2f}%\n")
        f.write(f"# Uniqueness: {metrics['uniqueness']:.2f}%\n")
        f.write(f"# Novelty: {metrics['novelty']:.2f}%\n")
        f.write(f"# Diversity: {metrics['diversity']:.2f}%\n")
        f.write(f"# Similarity: {metrics['similarity']:.2f}%\n")
        f.write(f"# FCD: {metrics['fcd']:.4f}\n")
        f.write(f"# Total valid molecules: {len(valid_smiles)}\n")
        f.write("#\n")
        for smi in valid_smiles:
            f.write(smi + '\n')
    
    print(f"\nSaved {len(valid_smiles)} valid molecules to {args.output}")
    
    # Save metrics to JSON
    metrics_file = args.output.replace('.txt', '_metrics.json')
    import json
    with open(metrics_file, 'w') as f:
        json.dump({k: float(v) if isinstance(v, (np.floating, float)) else v 
                   for k, v in metrics.items()}, f, indent=2)
    print(f"Saved metrics to {metrics_file}")
    
    # Generate visualization
    if args.visualize:
        viz_path = args.output.replace('.txt', '_pca.png')
        visualize_samples(z, viz_path, training_embeddings)

    # Print sample molecules
    if valid_smiles:
        print("\nSample generated molecules:")
        for i, smi in enumerate(valid_smiles[:10]):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                qed = QED.qed(mol)
                print(f"  {i+1}. {smi} (QED: {qed:.3f})")


if __name__ == "__main__":
    main()


# Example usage:
#
# Basic generation with metrics:
# python sample_from_prior.py \
#     --checkpoint checkpoints_dit/checkpoints_zinc_bbab_checkpoint_best.pt \
#     --grassy_checkpoint_dir outputs/BBAB_fixed_2026-01-27-16-11-28/ \
#     --training_smiles data/ZINC/BBAB/train.txt \
#     --dataset_name BBAB \
#     --n_latent_samples 1000 \
#     --output generated_bbab.txt
#
# With visualization:
# python sample_from_prior.py \
#     --checkpoint checkpoints_dit/checkpoints_zinc_bbab_checkpoint_best.pt \
#     --grassy_checkpoint_dir outputs/BBAB_fixed_2026-01-27-16-11-28/ \
#     --training_smiles data/ZINC/BBAB/train.txt \
#     --dataset_name BBAB \
#     --n_latent_samples 1000 \
#     --visualize \
#     --output generated_bbab.txt

# python -u -m grassy_dit.sample_unconstrained \
#     --checkpoint checkpoints/zinc/bbab/checkpoint_best.pt \
#     --grassy_checkpoint_dir outputs/BBAB_fixed_2026-01-28-01-25-28 \
#     --n_latent_samples 1000 \
#     --seed 42 \
#     --visualize \
#     --output generated_prior_seed42.txt

# For FBAB dataset:
# python sample_from_prior.py \
#     --checkpoint checkpoints_dit/checkpoints_zinc_fbab_checkpoint_best.pt \
#     --grassy_checkpoint_dir outputs/FBAB_fixed_2026-01-27-16-11-28/ \
#     --training_smiles data/ZINC/FBAB/train.txt \
#     --dataset_name FBAB \
#     --n_latent_samples 1000 \
#     --output generated_fbab.txt
#
# For JBCD dataset:
# python sample_from_prior.py \
#     --checkpoint checkpoints_dit/checkpoints_zinc_jbcd_checkpoint_best.pt \
#     --grassy_checkpoint_dir outputs/JBCD_fixed_2026-01-27-16-11-28/ \
#     --training_smiles data/ZINC/JBCD/train.txt \
#     --dataset_name JBCD \
#     --n_latent_samples 1000 \
#     --output generated_jbcd.txt


# python sample_from_prior.py \
#     --checkpoint checkpoints_dit/checkpoints_zinc_bbab_checkpoint_best.pt \
#     --grassy_checkpoint_dir outputs/BBAB_fixed_2026-01-27-16-11-28/ \
#     --n_latent_samples 100 \
#     --seed 42 \
#     --visualize \
#     --output generated_prior_seed42.txt

# python -m grassy_dit.sample_unconstrained \
#     --checkpoint checkpoints/zinc/bbab/checkpoint_best.pt \
#     --grassy_checkpoint_dir outputs/BBAB_fixed_2026-01-28-01-25-28 \
#     --n_latent_samples 10 \
#     --seed 42 \
#     --visualize \
