"""
CLI for scattering moment-guided molecule generation.

Usage:
    python -m guided_diffusion.sample \
        --checkpoint path/to/graphdit_checkpoint.pt \
        --target_smiles "CCO" \
        --guidance_scale 1.0 \
        --num_samples 32 \
        --output generated.txt

    # Or with target moments from file:
    python -m guided_diffusion.sample \
        --checkpoint path/to/graphdit_checkpoint.pt \
        --target_moments path/to/moments.npy \
        --guidance_scale 1.0 \
        --num_samples 32

    # Compare guided vs unguided:
    python -m guided_diffusion.sample \
        --checkpoint path/to/graphdit_checkpoint.pt \
        --target_smiles "CCO" \
        --compare_unguided \
        --num_samples 32
"""

import argparse
import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from guided_diffusion.generator import GuidedGraphDIT
from models.ScatteringTransform import GraphScatteringTransform
from evaluation.utils import (
    smiles_to_pyg_data,
    compute_scattering_from_smiles,
)


def load_graphdit_checkpoint(checkpoint_path: str, device: str = "cuda") -> GuidedGraphDIT:
    """
    Load a trained GraphDIT model from torch-molecule checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint (.pt file)
        device: Device to load on
    
    Returns:
        GuidedGraphDIT model ready for generation
    """
    model = GuidedGraphDIT()
    model.load_from_local(checkpoint_path)
    model.device = torch.device(device)
    return model


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
        # Use model's atom decoder if provided, otherwise fallback to generic list
        if model_atom_decoder is not None:
            atom_types = model_atom_decoder
            num_atom_types = len(atom_types)
        else:
            atom_types = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"][:num_atom_types]
        
        # Compute from SMILES
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
        model_atom_decoder: Atom type list from model (e.g. ['C', 'N', 'O', 'F'])
    
    Returns:
        Array of MSE distances [N]
    """
    # Use model's atom decoder if provided, otherwise fallback
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
        description='Generate molecules with scattering moment guidance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Model arguments
    parser.add_argument('--checkpoint', required=True,
                        help='Path to GraphDIT checkpoint (.pt file)')
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
    parser.add_argument('--num_samples', type=int, default=32,
                        help='Number of molecules to generate')
    parser.add_argument('--num_nodes', type=int, default=None,
                        help='Number of atoms per molecule (None = sample from training)')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                        help='Guidance scale (higher = stronger guidance)')
    parser.add_argument('--guidance_start_step', type=int, default=0,
                        help='Timestep to start applying guidance')
    
    # Scattering parameters
    parser.add_argument('--num_atom_types', type=int, default=None,
                        help='Number of atom types for scattering (auto-detected from model if not set)')
    parser.add_argument('--J', type=int, default=4,
                        help='Number of wavelet scales')
    parser.add_argument('--num_moments', type=int, default=4,
                        help='Number of statistical moments')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='guided_generated.txt',
                        help='Output file for generated SMILES')
    parser.add_argument('--compare_unguided', action='store_true',
                        help='Also generate unguided samples for comparison')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed progress')
    
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    
    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    model = load_graphdit_checkpoint(args.checkpoint, args.device)
    print("Model loaded successfully!")
    
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
    
    # Generate with guidance
    print(f"\nGenerating {args.num_samples} molecules with guidance_scale={args.guidance_scale}...")
    guided_smiles = model.guided_generate(
        target_moments=target_moments,
        num_nodes=args.num_nodes,
        batch_size=args.num_samples,
        guidance_scale=args.guidance_scale,
        guidance_start_step=args.guidance_start_step,
        num_atom_types=args.num_atom_types,
        J=args.J,
        num_moments=args.num_moments,
    )
    
    valid_guided = [s for s in guided_smiles if s is not None]
    print(f"Valid guided molecules: {len(valid_guided)}/{len(guided_smiles)}")
    
    # Compute moment distances for guided samples
    guided_distances = compute_moment_distances(
        guided_smiles, target_moments,
        args.num_atom_types, args.J, args.num_moments, args.device,
        model_atom_decoder=atom_decoder,
    )
    
    if len(guided_distances) > 0:
        print(f"Guided moment distance: mean={guided_distances.mean():.4f}, "
              f"std={guided_distances.std():.4f}, min={guided_distances.min():.4f}")
    
    # Optionally compare with unguided generation
    if args.compare_unguided:
        print(f"\nGenerating {args.num_samples} unguided molecules for comparison...")
        # Convert num_nodes to tensor for parent's generate
        num_nodes_tensor = None
        if args.num_nodes is not None:
            num_nodes_tensor = torch.tensor([[args.num_nodes]] * args.num_samples, device=args.device)
        unguided_smiles = model.generate(
            num_nodes=num_nodes_tensor,
            batch_size=args.num_samples,
        )
        
        valid_unguided = [s for s in unguided_smiles if s is not None]
        print(f"Valid unguided molecules: {len(valid_unguided)}/{len(unguided_smiles)}")
        
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
        f.write(f"# Scattering Moment-Guided Generation\n")
        f.write(f"# Checkpoint: {args.checkpoint}\n")
        if args.target_smiles:
            f.write(f"# Target SMILES: {args.target_smiles}\n")
        else:
            f.write(f"# Target moments: {args.target_moments}\n")
        f.write(f"# Guidance scale: {args.guidance_scale}\n")
        f.write(f"# Valid: {len(valid_guided)}/{len(guided_smiles)}\n")
        if len(guided_distances) > 0:
            f.write(f"# Mean moment distance: {guided_distances.mean():.4f}\n")
        f.write("#\n")
        for smi in valid_guided:
            f.write(f"{smi}\n")
    
    print(f"Done! Generated {len(valid_guided)} valid molecules.")
    
    # Print sample molecules
    if args.verbose and len(valid_guided) > 0:
        print("\nSample generated molecules:")
        for i, smi in enumerate(valid_guided[:5]):
            print(f"  {i+1}. {smi}")


if __name__ == "__main__":
    main()
