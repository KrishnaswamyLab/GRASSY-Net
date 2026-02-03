"""
Test script to verify scattering moment guidance and scaffold functions work correctly.

This script:
1. Tests that DenseSoftScattering produces differentiable gradients
2. Tests that guidance gradients point in the right direction
3. Tests that scaffold function correctly identifies fixed atoms
4. Tests the full guided generation pipeline (if checkpoint available)

Usage:
    python -m guided_diffusion.test_guidance
    python -m guided_diffusion.test_guidance --checkpoint path/to/model.pt
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_soft_scattering_differentiable():
    """Test that DenseSoftScattering produces gradients."""
    print("\n" + "="*60)
    print("Test 1: DenseSoftScattering differentiability")
    print("="*60)
    
    from grassy_dit.soft_scattering import DenseSoftScattering
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create scattering module
    num_atom_types = 10
    J = 4
    num_moments = 4
    scattering = DenseSoftScattering(num_atom_types, J, num_moments).to(device)
    
    # Create dummy soft graph
    B, N = 2, 15
    E = 5  # edge types
    
    soft_X = torch.randn(B, N, num_atom_types, device=device, requires_grad=True)
    soft_X = F.softmax(soft_X, dim=-1)  # Make valid probabilities
    soft_X.retain_grad()  # Keep gradients for non-leaf tensor
    
    soft_E = torch.randn(B, N, N, E, device=device, requires_grad=True)
    soft_E = F.softmax(soft_E, dim=-1)
    soft_E.retain_grad()  # Keep gradients for non-leaf tensor
    
    node_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    node_mask[:, 12:] = False  # Mask last 3 nodes
    
    # Forward pass
    moments = scattering(soft_X, soft_E, node_mask)
    
    print(f"Input soft_X shape: {soft_X.shape}")
    print(f"Input soft_E shape: {soft_E.shape}")
    print(f"Output moments shape: {moments.shape}")
    print(f"Expected shape: [B={B}, D={scattering.out_shape()}]")
    
    # Check gradients flow
    loss = moments.sum()
    loss.backward()
    
    has_grad_X = soft_X.grad is not None and soft_X.grad.abs().sum() > 0
    has_grad_E = soft_E.grad is not None and soft_E.grad.abs().sum() > 0
    
    print(f"\nGradient flows to soft_X: {has_grad_X}")
    print(f"Gradient flows to soft_E: {has_grad_E}")
    
    if has_grad_X and has_grad_E:
        print("✓ PASSED: Scattering is differentiable")
        return True
    else:
        print("✗ FAILED: No gradients!")
        return False


def test_guidance_direction():
    """Test that guidance gradients point toward target."""
    print("\n" + "="*60)
    print("Test 2: Guidance gradient direction")
    print("="*60)
    
    from guided_diffusion.guidance import ScatteringMomentGuidance
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create guidance module
    num_atom_types = 10
    J = 4
    num_moments = 4
    guidance = ScatteringMomentGuidance(num_atom_types, J, num_moments, device)
    
    # Create dummy predictions
    B, N = 2, 15
    E = 5
    
    pred_X = F.softmax(torch.randn(B, N, num_atom_types, device=device), dim=-1)
    pred_E = F.softmax(torch.randn(B, N, N, E, device=device), dim=-1)
    node_mask = torch.ones(B, N, dtype=torch.bool, device=device)
    node_mask[:, 12:] = False
    
    # Compute current moments
    current_moments = guidance.compute_moments(pred_X, pred_E, node_mask)
    print(f"Current moments shape: {current_moments.shape}")
    
    # Create target that's slightly different
    target_moments = current_moments + 0.5 * torch.randn_like(current_moments)
    
    # Get initial loss
    initial_loss = F.mse_loss(current_moments, target_moments).item()
    print(f"Initial MSE loss: {initial_loss:.4f}")
    
    # Compute guidance
    grad_X, grad_E, loss = guidance.compute_guidance(
        pred_X, pred_E, node_mask, target_moments, return_loss=True
    )
    
    print(f"Guidance grad_X shape: {grad_X.shape}")
    print(f"Guidance grad_E shape: {grad_E.shape}")
    print(f"Loss from guidance: {loss:.4f}")
    
    # Apply guidance (small step)
    # Note: guidance uses reduction='sum', so gradients are ~880x larger than mean
    step_size = 0.0001
    new_pred_X = F.softmax(torch.log(pred_X.clamp(min=1e-8)) + step_size * grad_X, dim=-1)
    new_pred_E = F.softmax(torch.log(pred_E.clamp(min=1e-8)) + step_size * grad_E, dim=-1)
    
    # Compute new moments
    new_moments = guidance.compute_moments(new_pred_X, new_pred_E, node_mask)
    new_loss = F.mse_loss(new_moments, target_moments).item()
    
    print(f"New MSE loss after gradient step: {new_loss:.4f}")
    print(f"Loss reduction: {initial_loss - new_loss:.4f}")
    
    if new_loss < initial_loss:
        print("✓ PASSED: Guidance reduces loss")
        return True
    else:
        print("✗ FAILED: Guidance did not reduce loss")
        return False


def test_scaffold_function():
    """Test that smiles_to_scaffold correctly identifies scaffold atoms."""
    print("\n" + "="*60)
    print("Test 3: Scaffold function")
    print("="*60)
    
    from guided_diffusion.generator import smiles_to_scaffold
    
    # Test molecule: toluene with amine (methylbenzene + NH2)
    full_smiles = "Cc1ccccc1N"  # 8 atoms: C(methyl) + 6 benzene + N
    max_nodes = 20
    atom_decoder = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"]
    
    print(f"Full molecule: {full_smiles}")
    print(f"Atom decoder: {atom_decoder[:4]}...")
    
    # Test 1: Keep benzene ring via SMARTS
    print("\n--- Test 3a: SMARTS pattern ---")
    scaffold_pattern = "c1ccccc1"  # benzene ring
    
    X, E, scaffold_mask, node_mask, n_atoms = smiles_to_scaffold(
        full_smiles=full_smiles,
        max_nodes=max_nodes,
        atom_decoder=atom_decoder,
        scaffold_pattern=scaffold_pattern,
    )
    
    print(f"Total atoms: {n_atoms}")
    print(f"X shape: {X.shape}")
    print(f"E shape: {E.shape}")
    print(f"Scaffold mask: {scaffold_mask[:n_atoms].tolist()}")
    print(f"Scaffold atoms: {scaffold_mask.sum().item()}")
    
    # Benzene has 6 atoms, so scaffold should have 6 True values
    expected_scaffold_count = 6
    actual_scaffold_count = scaffold_mask.sum().item()
    
    if actual_scaffold_count == expected_scaffold_count:
        print(f"✓ Correct: {actual_scaffold_count} scaffold atoms (benzene)")
    else:
        print(f"✗ FAILED: Expected {expected_scaffold_count}, got {actual_scaffold_count}")
        return False
    
    # Test 2: Remove specific atoms
    print("\n--- Test 3b: Remove indices ---")
    # Remove methyl (index 0) and amine (index 7) - keep benzene
    remove_indices = [0, 7]
    
    X2, E2, scaffold_mask2, node_mask2, n_atoms2 = smiles_to_scaffold(
        full_smiles=full_smiles,
        max_nodes=max_nodes,
        atom_decoder=atom_decoder,
        remove_indices=remove_indices,
    )
    
    print(f"Remove indices: {remove_indices}")
    print(f"Scaffold mask: {scaffold_mask2[:n_atoms2].tolist()}")
    print(f"Scaffold atoms: {scaffold_mask2.sum().item()}")
    
    # Should have n_atoms - len(remove_indices) = 8 - 2 = 6 scaffold atoms
    expected = n_atoms - len(remove_indices)
    actual = scaffold_mask2.sum().item()
    
    if actual == expected:
        print(f"✓ Correct: {actual} scaffold atoms (removed {len(remove_indices)})")
    else:
        print(f"✗ FAILED: Expected {expected}, got {actual}")
        return False
    
    # Test 3: Verify scaffold atoms have features, non-scaffold are zero
    print("\n--- Test 3c: Feature masking ---")
    scaffold_features = X[scaffold_mask[:max_nodes]].sum().item()
    non_scaffold_features = X[~scaffold_mask[:max_nodes]].sum().item()
    
    print(f"Scaffold atom features sum: {scaffold_features}")
    print(f"Non-scaffold atom features sum: {non_scaffold_features}")
    
    if scaffold_features > 0 and non_scaffold_features == 0:
        print("✓ Correct: Only scaffold atoms have features")
        print("✓ PASSED: Scaffold function works correctly")
        return True
    else:
        print("✗ FAILED: Feature masking incorrect")
        return False


def test_full_pipeline(checkpoint_path: str):
    """Test full guided generation pipeline."""
    print("\n" + "="*60)
    print("Test 4: Full guided generation pipeline")
    print("="*60)
    
    from guided_diffusion.generator import GuidedGraphDIT
    from guided_diffusion.sample import get_target_moments, compute_moment_distances
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model from {checkpoint_path}...")
    model = GuidedGraphDIT()
    try:
        model.load_from_local(checkpoint_path)
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Skipping full pipeline test.")
        return None
    
    model.device = torch.device(device)
    print("Model loaded!")
    
    # Test parameters
    target_smiles = "c1ccccc1"  # Benzene
    num_samples = 4
    num_nodes = 10
    
    print(f"\nTarget molecule: {target_smiles}")
    print(f"Generating {num_samples} samples with {num_nodes} nodes")
    
    # Get target moments
    target_moments = get_target_moments(
        target_smiles=target_smiles,
        num_atom_types=10,
        J=4,
        num_moments=4,
        device=device
    )
    print(f"Target moments shape: {target_moments.shape}")
    
    # Generate with guidance
    print("\nGenerating with guidance (scale=1.0)...")
    guided_smiles = model.guided_generate(
        target_moments=target_moments,
        num_nodes=num_nodes,
        batch_size=num_samples,
        guidance_scale=1.0,
        num_atom_types=10,
        J=4,
        num_moments=4,
    )
    
    valid_guided = [s for s in guided_smiles if s is not None]
    print(f"Valid guided: {len(valid_guided)}/{len(guided_smiles)}")
    
    if len(valid_guided) > 0:
        guided_distances = compute_moment_distances(
            guided_smiles, target_moments, 10, 4, 4, device
        )
        print(f"Guided mean distance: {guided_distances.mean():.4f}")
    
    # Generate without guidance
    print("\nGenerating without guidance...")
    unguided_smiles = model.generate(
        num_nodes=num_nodes,
        batch_size=num_samples,
    )
    
    valid_unguided = [s for s in unguided_smiles if s is not None]
    print(f"Valid unguided: {len(valid_unguided)}/{len(unguided_smiles)}")
    
    if len(valid_unguided) > 0:
        unguided_distances = compute_moment_distances(
            unguided_smiles, target_moments, 10, 4, 4, device
        )
        print(f"Unguided mean distance: {unguided_distances.mean():.4f}")
    
    # Compare
    if len(valid_guided) > 0 and len(valid_unguided) > 0:
        improvement = (unguided_distances.mean() - guided_distances.mean()) / unguided_distances.mean() * 100
        print(f"\nImprovement: {improvement:.1f}%")
        
        if improvement > 0:
            print("✓ PASSED: Guidance improves moment matching")
            return True
        else:
            print("✗ WARNING: Guidance did not improve moment matching")
            print("  (This can happen with small samples, try larger batch)")
            return False
    
    return None


def main():
    parser = argparse.ArgumentParser(description='Test scattering moment guidance and scaffold')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to GraphDIT checkpoint for full pipeline test')
    args = parser.parse_args()
    
    print("="*60)
    print("Scattering Moment Guidance - Test Suite")
    print("="*60)
    
    results = {}
    
    # Test 1: Differentiability
    results['differentiable'] = test_soft_scattering_differentiable()
    
    # Test 2: Gradient direction
    results['gradient_direction'] = test_guidance_direction()
    
    # Test 3: Scaffold function
    results['scaffold'] = test_scaffold_function()
    
    # Test 4: Full pipeline (optional)
    if args.checkpoint:
        results['full_pipeline'] = test_full_pipeline(args.checkpoint)
    else:
        print("\n" + "="*60)
        print("Test 4: Full pipeline (SKIPPED - no checkpoint provided)")
        print("="*60)
        print("To run full pipeline test, provide --checkpoint path/to/model.pt")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    for name, result in results.items():
        status = "✓ PASSED" if result is True else ("✗ FAILED" if result is False else "- SKIPPED")
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
