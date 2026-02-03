"""
Test if gradient direction is correct by comparing normal vs flipped gradients.
"""
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from guided_diffusion.generator import GuidedGraphDIT
from guided_diffusion.sample import get_target_moments, compute_moment_distances
from guided_diffusion.guidance import ScatteringMomentGuidance


class FlippedGuidance(ScatteringMomentGuidance):
    """Same as ScatteringMomentGuidance but with POSITIVE gradient (wrong direction)."""
    
    def compute_guidance(self, pred_X, pred_E, node_mask, target_moments, return_loss=False):
        pred_X = pred_X.detach().clone().requires_grad_(True)
        pred_E = pred_E.detach().clone().requires_grad_(True)
        
        current_moments = self.scattering(pred_X, pred_E, node_mask)
        loss = F.mse_loss(current_moments, target_moments, reduction='sum')
        loss.backward()
        
        # FLIPPED: Return POSITIVE gradients (WRONG direction - should INCREASE loss)
        grad_X = +pred_X.grad  # Changed from -pred_X.grad
        grad_E = +pred_E.grad  # Changed from -pred_E.grad
        
        if return_loss:
            return grad_X, grad_E, loss.item()
        return grad_X, grad_E


def test_direction():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = GuidedGraphDIT()
    model.load_from_local('runs/graphdit_qm9_43996065/graphdit_final.pt')
    model.device = torch.device(device)
    atom_decoder = model.dataset_info['atom_decoder']
    num_atom_types = len(atom_decoder)
    print(f"Atom types: {atom_decoder}")
    
    # Get target
    target = get_target_moments('c1ccccc1', model_atom_decoder=atom_decoder, device=device)
    print(f"Target shape: {target.shape}")
    
    num_samples = 50
    scale = 0.1
    end_step = 450
    
    # Test 1: Unguided baseline
    print(f"\n{'='*50}")
    print("Test 1: UNGUIDED baseline")
    print(f"{'='*50}")
    num_nodes_tensor = torch.tensor([[9]] * num_samples, device=device)
    smiles_baseline = model.generate(num_nodes=num_nodes_tensor, batch_size=num_samples)
    valid_baseline = [s for s in smiles_baseline if s]
    dist_baseline = compute_moment_distances(
        smiles_baseline, target, num_atom_types, 4, 4, device, atom_decoder
    )
    print(f"Valid: {len(valid_baseline)}/{num_samples}")
    print(f"Mean distance: {dist_baseline.mean():.4f}")
    
    # Test 2: Normal guidance (negative gradient = minimize loss)
    print(f"\n{'='*50}")
    print("Test 2: NORMAL gradient (-grad, should MINIMIZE loss)")
    print(f"{'='*50}")
    smiles_normal = model.guided_generate(
        target_moments=target,
        num_nodes=9,
        batch_size=num_samples,
        guidance_scale=scale,
        guidance_end_step=end_step,
    )
    valid_normal = [s for s in smiles_normal if s]
    dist_normal = compute_moment_distances(
        smiles_normal, target, num_atom_types, 4, 4, device, atom_decoder
    )
    print(f"Valid: {len(valid_normal)}/{num_samples}")
    print(f"Mean distance: {dist_normal.mean():.4f}")
    
    # Test 3: Flipped guidance (positive gradient = maximize loss)
    print(f"\n{'='*50}")
    print("Test 3: FLIPPED gradient (+grad, should MAXIMIZE loss)")
    print(f"{'='*50}")
    
    # Temporarily replace guidance class
    from guided_diffusion import guidance as guidance_module
    original_class = guidance_module.ScatteringMomentGuidance
    guidance_module.ScatteringMomentGuidance = FlippedGuidance
    
    # Need to reimport generator to pick up the change
    import importlib
    from guided_diffusion import generator as gen_module
    importlib.reload(gen_module)
    
    model_flipped = gen_module.GuidedGraphDIT()
    model_flipped.load_from_local('runs/graphdit_qm9_43996065/graphdit_final.pt')
    model_flipped.device = torch.device(device)
    
    smiles_flipped = model_flipped.guided_generate(
        target_moments=target,
        num_nodes=9,
        batch_size=num_samples,
        guidance_scale=scale,
        guidance_end_step=end_step,
    )
    valid_flipped = [s for s in smiles_flipped if s]
    dist_flipped = compute_moment_distances(
        smiles_flipped, target, num_atom_types, 4, 4, device, atom_decoder
    )
    print(f"Valid: {len(valid_flipped)}/{num_samples}")
    print(f"Mean distance: {dist_flipped.mean():.4f}")
    
    # Restore original class
    guidance_module.ScatteringMomentGuidance = original_class
    
    # Summary
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"{'Method':<20} {'Valid':<10} {'Mean Dist':<12}")
    print("-"*50)
    print(f"{'Unguided':<20} {len(valid_baseline):<10} {dist_baseline.mean():<12.4f}")
    print(f"{'Normal (-grad)':<20} {len(valid_normal):<10} {dist_normal.mean():<12.4f}")
    print(f"{'Flipped (+grad)':<20} {len(valid_flipped):<10} {dist_flipped.mean():<12.4f}")
    
    print("\nInterpretation:")
    if dist_normal.mean() < dist_baseline.mean():
        print("✓ Normal gradient HELPS (reduces distance)")
    else:
        print("✗ Normal gradient HURTS (increases distance)")
    
    if dist_flipped.mean() > dist_normal.mean():
        print("✓ Flipped is WORSE than normal → gradient direction is CORRECT")
    else:
        print("✗ Flipped is BETTER than normal → gradient direction is WRONG!")


if __name__ == "__main__":
    test_direction()
