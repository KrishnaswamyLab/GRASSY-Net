"""
Diagnostic script for Soft Scattering Guidance.

This script runs Phase 1 diagnostics to understand why soft scattering guidance
may be failing during inference. It logs detailed information about:
- Gradient magnitudes vs prediction magnitudes
- The ratio of guidance signal to model predictions
- How the posterior computation affects the guidance signal

Usage:
    python -m guided_diffusion.diagnose_guidance \
        --checkpoint path/to/graphdit_checkpoint.pt \
        --target_smiles "CCO" \
        --guidance_scale 1.0 \
        --num_samples 4

    # Compare with no guidance:
    python -m guided_diffusion.diagnose_guidance \
        --checkpoint path/to/graphdit_checkpoint.pt \
        --target_smiles "CCO" \
        --guidance_scale 0.0 \
        --num_samples 4
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from guided_diffusion.generator import GuidedGraphDIT
from guided_diffusion.sample import get_target_moments, compute_moment_distances


def run_diagnostic(
    checkpoint_path: str,
    target_smiles: str,
    guidance_scale: float = 1.0,
    num_samples: int = 4,
    num_nodes: int = 10,
    device: str = "cuda",
    guidance_start_step: int = 0,
):
    """
    Run diagnostic generation with detailed logging.

    This function enables diagnostic logging in the GuidedGraphDIT model
    and runs a small generation to capture key metrics at early, middle,
    and late diffusion steps.
    """
    print("=" * 70)
    print("SOFT SCATTERING GUIDANCE DIAGNOSTIC")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  - Checkpoint: {checkpoint_path}")
    print(f"  - Target SMILES: {target_smiles}")
    print(f"  - Guidance scale: {guidance_scale}")
    print(f"  - Num samples: {num_samples}")
    print(f"  - Num nodes: {num_nodes}")
    print(f"  - Device: {device}")
    print(f"  - Guidance start step: {guidance_start_step}")

    # Load model
    print(f"\nLoading model...")
    model = GuidedGraphDIT()
    model.load_from_local(checkpoint_path)
    model.device = torch.device(device)

    # Enable diagnostic logging
    model._diagnostic_logging = True

    # Get atom decoder
    atom_decoder = None
    num_atom_types = 10
    if hasattr(model, 'dataset_info') and model.dataset_info:
        atom_decoder = model.dataset_info.get('atom_decoder', None)
        if atom_decoder:
            num_atom_types = len(atom_decoder)
            print(f"  - Atom types ({num_atom_types}): {atom_decoder}")

    # Compute target moments
    print(f"\nComputing target moments for: {target_smiles}")
    target_moments = get_target_moments(
        target_smiles=target_smiles,
        num_atom_types=num_atom_types,
        J=4,
        num_moments=4,
        device=device,
        model_atom_decoder=atom_decoder,
    )
    print(f"Target moments shape: {target_moments.shape}")
    print(f"Target moments norm: {target_moments.norm().item():.4f}")
    print(f"Target moments range: [{target_moments.min().item():.4f}, {target_moments.max().item():.4f}]")

    # Run guided generation with diagnostics
    print("\n" + "=" * 70)
    print(f"RUNNING GUIDED GENERATION (scale={guidance_scale})")
    print("=" * 70)
    print("\nDiagnostic output will appear below at steps: early, middle, late")
    print("-" * 70)

    guided_smiles = model.guided_generate(
        target_moments=target_moments,
        num_nodes=num_nodes,
        batch_size=num_samples,
        guidance_scale=guidance_scale,
        guidance_start_step=guidance_start_step,
        num_atom_types=num_atom_types,
        J=4,
        num_moments=4,
    )

    print("-" * 70)

    # Report results
    valid_guided = [s for s in guided_smiles if s is not None]
    print(f"\n\nRESULTS:")
    print(f"  Valid molecules: {len(valid_guided)}/{len(guided_smiles)}")

    if len(valid_guided) > 0:
        guided_distances = compute_moment_distances(
            guided_smiles, target_moments,
            num_atom_types, 4, 4, device,
            model_atom_decoder=atom_decoder,
        )
        print(f"  Moment distances: mean={guided_distances.mean():.4f}, min={guided_distances.min():.4f}")
        print(f"\n  Generated molecules:")
        for i, smi in enumerate(valid_guided[:5]):
            print(f"    {i+1}. {smi}")

    return guided_smiles, valid_guided


def interpret_diagnostics():
    """Print interpretation guide for diagnostic output."""
    print("\n" + "=" * 70)
    print("HOW TO INTERPRET DIAGNOSTIC OUTPUT")
    print("=" * 70)
    print("""
CHECK A: Is ratio_X consistently below 0.01?
  -> If YES: Guidance gradient is negligible. The guidance is doing nothing.
     FIX: Apply MOOD-style gradient normalization.

CHECK B: Is ratio_X reasonable (0.01 - 0.5) but "After posterior" values
         are nearly identical to "Before guidance" values?
  -> If YES: Posterior computation washes out the guidance signal.
     FIX: Apply guidance AFTER the posterior computation.

CHECK C: Is moment L2 distance similar at early and late steps?
  -> If YES at early but NO at late: Guidance only useful at late steps.
     FIX: Restrict guidance to last 30-50% of steps.
  -> If YES at all steps: Scattering may not differentiate on soft graphs.

CHECK D: Is ratio_X very large (above 1.0)?
  -> If YES: Guidance is overpowering the model, breaking chemistry.
     FIX: Apply MOOD-style normalization to prevent this.
""")


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose soft scattering guidance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--checkpoint', required=True,
                        help='Path to GraphDIT checkpoint')
    parser.add_argument('--target_smiles', default='CCO',
                        help='Target molecule SMILES (default: ethanol)')
    parser.add_argument('--guidance_scale', type=float, default=1.0,
                        help='Guidance scale (try 0.0, 1.0, 10.0)')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of molecules to generate')
    parser.add_argument('--num_nodes', type=int, default=10,
                        help='Number of atoms per molecule')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (cuda/cpu)')
    parser.add_argument('--guidance_start_step', type=int, default=0,
                        help='Step to start applying guidance (e.g., 250 for last 50%%)')
    parser.add_argument('--interpret', action='store_true',
                        help='Just print interpretation guide')

    args = parser.parse_args()

    if args.interpret:
        interpret_diagnostics()
        return 0

    # Run diagnostic
    run_diagnostic(
        checkpoint_path=args.checkpoint,
        target_smiles=args.target_smiles,
        guidance_scale=args.guidance_scale,
        num_samples=args.num_samples,
        num_nodes=args.num_nodes,
        device=args.device,
        guidance_start_step=args.guidance_start_step,
    )

    # Print interpretation guide
    interpret_diagnostics()

    return 0


if __name__ == "__main__":
    sys.exit(main())
