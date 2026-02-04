"""
CLI for classifier-guided molecule generation.

Generates molecules using a trained DiT model with guidance from a
trained moment classifier to steer toward target scattering moments.

Usage:
    # Generate with target SMILES
    python -m guided_diffusion.generate_classifier_guided \
        --dit_checkpoint path/to/dit.pt \
        --classifier_checkpoint path/to/classifier.pt \
        --target_smiles "c1ccccc1" \
        --guidance_scale 1.0 \
        --num_samples 100 \
        --output generated.txt

    # Generate with target moments from file
    python -m guided_diffusion.generate_classifier_guided \
        --dit_checkpoint path/to/dit.pt \
        --classifier_checkpoint path/to/classifier.pt \
        --target_moments path/to/moments.npy \
        --guidance_scale 1.0 \
        --num_samples 100

    # Compare with unguided generation
    python -m guided_diffusion.generate_classifier_guided \
        --dit_checkpoint path/to/dit.pt \
        --classifier_checkpoint path/to/classifier.pt \
        --target_smiles "CCO" \
        --compare_unguided \
        --num_samples 100
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from guided_diffusion.classifier_generator import (
    ClassifierGuidedGraphDIT,
    load_classifier_guided_model,
)
from guided_diffusion.classifier_guidance import ClassifierGuidance
from models.ScatteringTransform import GraphScatteringTransform
from evaluation.utils import compute_scattering_from_smiles


def get_target_moments(
    target_smiles: Optional[str] = None,
    target_moments_path: Optional[str] = None,
    target_index: Optional[int] = None,
    num_atom_types: int = 10,
    J: int = 4,
    num_moments: int = 4,
    device: str = "cuda",
    model_atom_decoder: Optional[List[str]] = None,
) -> torch.Tensor:
    """
    Get target scattering moments from SMILES or file.
    
    Args:
        target_smiles: Reference molecule SMILES
        target_moments_path: Path to .npy file with precomputed moments
        target_index: Index in moments file (if multiple)
        num_atom_types: Number of atom types
        J: Number of wavelet scales
        num_moments: Number of statistical moments
        device: Device for computation
        model_atom_decoder: Atom type list from model (e.g. ['C', 'N', 'O', 'F'])
    
    Returns:
        Target moments tensor [D]
    """
    if target_smiles is not None:
        # Use model's atom decoder if provided
        if model_atom_decoder is not None:
            atom_types = model_atom_decoder
            num_atom_types = len(atom_types)
        else:
            atom_types = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"][:num_atom_types]
        
        # Compute scattering from SMILES
        scattering_model = GraphScatteringTransform(
            in_channels=num_atom_types, J=J, num_moments=num_moments
        ).to(device)
        scattering_model.eval()
        
        moments = compute_scattering_from_smiles(
            [target_smiles], scattering_model, atom_types, device
        )
        
        if moments.shape[0] == 0:
            raise ValueError(f"Could not compute scattering for SMILES: {target_smiles}")
        
        return moments[0].to(device)
    
    elif target_moments_path is not None:
        # Load from file
        moments = np.load(target_moments_path)
        
        if moments.ndim == 2:
            idx = target_index if target_index is not None else 0
            moments = moments[idx]
        
        return torch.from_numpy(moments).float().to(device)
    
    else:
        raise ValueError("Must provide either target_smiles or target_moments_path")


def compute_moment_distances(
    smiles_list: List[str],
    target_moments: torch.Tensor,
    num_atom_types: int = 10,
    J: int = 4,
    num_moments: int = 4,
    device: str = "cuda",
    model_atom_decoder: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute MSE distances between generated molecules and target moments.
    
    Args:
        smiles_list: List of generated SMILES
        target_moments: Target scattering moments [D]
        num_atom_types: Number of atom types
        J: Number of wavelet scales
        num_moments: Number of statistical moments
        device: Device for computation
        model_atom_decoder: Atom type list from model
    
    Returns:
        Array of MSE distances [N]
    """
    if model_atom_decoder is not None:
        atom_types = model_atom_decoder
        num_atom_types = len(atom_types)
    else:
        atom_types = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"][:num_atom_types]
    
    scattering_model = GraphScatteringTransform(
        in_channels=num_atom_types, J=J, num_moments=num_moments
    ).to(device)
    scattering_model.eval()
    
    # Filter valid SMILES
    valid_smiles = [s for s in smiles_list if s is not None]
    
    if len(valid_smiles) == 0:
        return np.array([])
    
    # Compute scattering for generated molecules
    gen_moments = compute_scattering_from_smiles(
        valid_smiles, scattering_model, atom_types, device
    )
    
    # Compute MSE distances
    target_expanded = target_moments.unsqueeze(0).expand(gen_moments.shape[0], -1)
    distances = ((gen_moments.to(device) - target_expanded) ** 2).mean(dim=-1)
    
    return distances.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description='Generate molecules with classifier-based moment guidance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model arguments
    parser.add_argument('--dit_checkpoint', required=True,
                        help='Path to DiT model checkpoint (.pt file)')
    parser.add_argument('--classifier_checkpoint', required=True,
                        help='Path to trained moment classifier checkpoint (.pt file)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda/cpu)')
    
    # Target specification
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument('--target_smiles', type=str,
                              help='Reference molecule SMILES for target moments')
    target_group.add_argument('--target_moments', type=str,
                              help='Path to .npy file with target moments')
    parser.add_argument('--target_index', type=int, default=None,
                        help='Index in moments file (if multiple)')
    
    # Generation arguments
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of molecules to generate')
    parser.add_argument('--num_nodes', type=int, default=None,
                        help='Number of atoms per molecule (None = sample from training)')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                        help='Guidance scale (higher = stronger guidance)')
    parser.add_argument('--guidance_start_step', type=int, default=0,
                        help='Timestep to start applying guidance')
    parser.add_argument('--guidance_end_step', type=int, default=None,
                        help='Timestep to stop applying guidance (None = guide until end)')
    
    # Scattering parameters (for computing target and evaluating)
    parser.add_argument('--num_atom_types', type=int, default=None,
                        help='Number of atom types (auto-detected from model if not set)')
    parser.add_argument('--J', type=int, default=4,
                        help='Number of wavelet scales')
    parser.add_argument('--num_moments', type=int, default=4,
                        help='Number of statistical moments')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='classifier_guided_generated.txt',
                        help='Output file for generated SMILES')
    parser.add_argument('--compare_unguided', action='store_true',
                        help='Also generate unguided samples for comparison')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Load DiT model
    print(f"\nLoading DiT model from {args.dit_checkpoint}...")
    model = load_classifier_guided_model(args.dit_checkpoint, args.device)
    print("DiT model loaded!")
    
    # Auto-detect atom types from model
    atom_decoder = None
    if hasattr(model, 'dataset_info') and model.dataset_info:
        atom_decoder = model.dataset_info.get('atom_decoder', None)
        if atom_decoder:
            args.num_atom_types = len(atom_decoder)
            print(f"Auto-detected {args.num_atom_types} atom types: {atom_decoder}")
    
    if atom_decoder is None:
        if args.num_atom_types is None:
            args.num_atom_types = 10
        print(f"Using fallback: {args.num_atom_types} atom types")
    
    # Load classifier
    print(f"\nLoading classifier from {args.classifier_checkpoint}...")
    classifier_guidance = ClassifierGuidance(
        classifier=args.classifier_checkpoint,
        device=args.device,
    )
    print(f"Classifier loaded! Moment dim: {classifier_guidance.moment_dim}")
    
    # Get target moments
    print("\nComputing target moments...")
    target_moments = get_target_moments(
        target_smiles=args.target_smiles,
        target_moments_path=args.target_moments,
        target_index=args.target_index,
        num_atom_types=args.num_atom_types,
        J=args.J,
        num_moments=args.num_moments,
        device=args.device,
        model_atom_decoder=atom_decoder,
    )
    print(f"Target moments shape: {target_moments.shape}")
    
    # Generate with classifier guidance
    print(f"\nGenerating {args.num_samples} molecules with classifier guidance...")
    print(f"  Guidance scale: {args.guidance_scale}")
    print(f"  Guidance window: step {args.guidance_start_step} to {args.guidance_end_step or 'end'}")
    
    guided_smiles = model.classifier_guided_generate_with_preloaded(
        classifier_guidance=classifier_guidance,
        target_moments=target_moments,
        num_nodes=args.num_nodes,
        batch_size=args.num_samples,
        guidance_scale=args.guidance_scale,
        guidance_start_step=args.guidance_start_step,
        guidance_end_step=args.guidance_end_step,
    )
    
    valid_guided = [s for s in guided_smiles if s is not None]
    print(f"Valid molecules: {len(valid_guided)}/{len(guided_smiles)} ({100*len(valid_guided)/len(guided_smiles):.1f}%)")
    
    # Compute moment distances
    guided_distances = compute_moment_distances(
        guided_smiles, target_moments,
        args.num_atom_types, args.J, args.num_moments, args.device,
        model_atom_decoder=atom_decoder,
    )
    
    if len(guided_distances) > 0:
        print(f"Moment distance: mean={guided_distances.mean():.4f}, "
              f"std={guided_distances.std():.4f}, min={guided_distances.min():.4f}")
    
    # Compare with unguided generation if requested
    if args.compare_unguided:
        print(f"\nGenerating {args.num_samples} unguided molecules for comparison...")
        
        num_nodes_tensor = None
        if args.num_nodes is not None:
            num_nodes_tensor = torch.tensor([[args.num_nodes]] * args.num_samples, device=args.device)
        
        unguided_smiles = model.generate(
            num_nodes=num_nodes_tensor,
            batch_size=args.num_samples,
        )
        
        valid_unguided = [s for s in unguided_smiles if s is not None]
        print(f"Valid unguided: {len(valid_unguided)}/{len(unguided_smiles)} ({100*len(valid_unguided)/len(unguided_smiles):.1f}%)")
        
        unguided_distances = compute_moment_distances(
            unguided_smiles, target_moments,
            args.num_atom_types, args.J, args.num_moments, args.device,
            model_atom_decoder=atom_decoder,
        )
        
        if len(unguided_distances) > 0:
            print(f"Unguided moment distance: mean={unguided_distances.mean():.4f}, "
                  f"std={unguided_distances.std():.4f}, min={unguided_distances.min():.4f}")
        
        # Report improvement
        if len(guided_distances) > 0 and len(unguided_distances) > 0:
            improvement = (unguided_distances.mean() - guided_distances.mean()) / unguided_distances.mean() * 100
            print(f"\nImprovement: {improvement:.1f}% reduction in moment distance")
    
    # Save results
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w') as f:
        f.write(f"# Classifier-Guided Generation\n")
        f.write(f"# DiT checkpoint: {args.dit_checkpoint}\n")
        f.write(f"# Classifier checkpoint: {args.classifier_checkpoint}\n")
        if args.target_smiles:
            f.write(f"# Target SMILES: {args.target_smiles}\n")
        else:
            f.write(f"# Target moments: {args.target_moments}\n")
        f.write(f"# Guidance scale: {args.guidance_scale}\n")
        f.write(f"# Valid: {len(valid_guided)}/{len(guided_smiles)}\n")
        if len(guided_distances) > 0:
            f.write(f"# Mean moment distance: {guided_distances.mean():.4f}\n")
            f.write(f"# Min moment distance: {guided_distances.min():.4f}\n")
        f.write("#\n")
        for smi in valid_guided:
            f.write(f"{smi}\n")
    
    print(f"Done! Generated {len(valid_guided)} valid molecules.")
    
    # Print sample molecules
    if args.verbose and len(valid_guided) > 0:
        print("\nSample generated molecules:")
        for i, smi in enumerate(valid_guided[:5]):
            if len(guided_distances) > i:
                print(f"  {i+1}. {smi} (dist: {guided_distances[i]:.4f})")
            else:
                print(f"  {i+1}. {smi}")


if __name__ == "__main__":
    main()
