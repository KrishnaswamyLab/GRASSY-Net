"""
Sweep guidance scales to find optimal value for scattering moment guidance.

Usage:
    python -m guided_diffusion.sweep_guidance_scale
    python -m guided_diffusion.sweep_guidance_scale --checkpoint path/to/model.pt --target "CCO"
"""
import argparse
import torch
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from guided_diffusion.generator import GuidedGraphDIT
from guided_diffusion.sample import get_target_moments, compute_moment_distances


def main():
    parser = argparse.ArgumentParser(description='Sweep guidance scales')
    parser.add_argument('--checkpoint', type=str, 
                        default='runs/graphdit_qm9_43996065/graphdit_final.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--target', type=str, default='c1ccccc1',
                        help='Target SMILES for moment matching')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples per scale')
    parser.add_argument('--num_nodes', type=int, default=9,
                        help='Number of atoms per molecule')
    parser.add_argument('--scales', type=float, nargs='+',
                        default=[0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
                        help='Guidance scales to test (0.0 = unguided baseline)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    args = parser.parse_args()

    print("="*70)
    print("Guidance Scale Sweep")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Target SMILES: {args.target}")
    print(f"Num samples: {args.num_samples}")
    print(f"Num nodes: {args.num_nodes}")
    print(f"Scales: {args.scales}")
    print(f"Device: {args.device}")
    print("="*70 + "\n")

    # Load model
    print("Loading model...")
    model = GuidedGraphDIT()
    model.load_from_local(args.checkpoint)
    model.device = torch.device(args.device)
    atom_decoder = model.dataset_info['atom_decoder']
    num_atom_types = len(atom_decoder)
    print(f"Atom types ({num_atom_types}): {atom_decoder}\n")

    # Get target moments
    print("Computing target moments...")
    target = get_target_moments(
        target_smiles=args.target,
        model_atom_decoder=atom_decoder,
        device=args.device
    )
    print(f"Target moments shape: {target.shape}\n")

    # Results storage
    results = []

    for scale in args.scales:
        print(f"\n{'='*50}")
        print(f"Guidance scale: {scale}")
        print(f"{'='*50}")
        
        if scale == 0.0:
            # Unguided baseline
            num_nodes_tensor = torch.tensor([[args.num_nodes]] * args.num_samples, 
                                            device=args.device)
            smiles = model.generate(num_nodes=num_nodes_tensor, 
                                   batch_size=args.num_samples)
        else:
            smiles = model.guided_generate(
                target_moments=target,
                num_nodes=args.num_nodes,
                batch_size=args.num_samples,
                guidance_scale=scale,
            )
        
        valid = [s for s in smiles if s is not None]
        validity = len(valid) / len(smiles) * 100
        
        distances = compute_moment_distances(
            smiles, target, num_atom_types, 4, 4, args.device,
            model_atom_decoder=atom_decoder
        )
        
        if len(distances) > 0:
            mean_dist = float(distances.mean())
            std_dist = float(distances.std())
            min_dist = float(distances.min())
        else:
            mean_dist = std_dist = min_dist = float('nan')
        
        print(f"Valid: {len(valid)}/{len(smiles)} ({validity:.1f}%)")
        print(f"Distance: mean={mean_dist:.4f}, std={std_dist:.4f}, min={min_dist:.4f}")
        
        # Show sample molecules
        print(f"Samples: {valid[:3]}")
        
        results.append({
            'scale': scale,
            'validity': validity,
            'mean_dist': mean_dist,
            'std_dist': std_dist,
            'min_dist': min_dist,
            'n_valid': len(valid),
        })

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Scale':<10} {'Validity':<12} {'Mean Dist':<12} {'Std':<12} {'Min':<12}")
    print("-"*70)
    
    baseline_dist = None
    for r in results:
        if r['scale'] == 0.0:
            baseline_dist = r['mean_dist']
        print(f"{r['scale']:<10.3f} {r['validity']:<12.1f} {r['mean_dist']:<12.4f} "
              f"{r['std_dist']:<12.4f} {r['min_dist']:<12.4f}")
    
    # Best result
    print("-"*70)
    valid_results = [r for r in results if r['validity'] > 50 and r['mean_dist'] == r['mean_dist']]
    if valid_results:
        best = min(valid_results, key=lambda x: x['mean_dist'])
        print(f"\nBest scale: {best['scale']} (mean_dist={best['mean_dist']:.4f})")
        
        if baseline_dist and baseline_dist == baseline_dist:
            improvement = (baseline_dist - best['mean_dist']) / baseline_dist * 100
            print(f"Improvement over unguided: {improvement:.1f}%")


if __name__ == "__main__":
    main()
