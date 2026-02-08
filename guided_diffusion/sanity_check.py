"""Quick local sanity check for guidance math. No checkpoint needed."""
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')

from guided_diffusion.guidance import ScatteringMomentGuidance, apply_guidance_to_probs
from guided_diffusion.edge_guidance import guided_edge_step, compute_expected_bond_count

device = 'cpu'
B, N, V, E = 2, 10, 10, 5
guidance = ScatteringMomentGuidance(num_atom_types=V, J=4, num_moments=4, device=device)

pred_X = F.softmax(torch.randn(B, N, V), dim=-1)
pred_E = F.softmax(torch.randn(B, N, N, E), dim=-1)
node_mask = torch.ones(B, N, dtype=torch.bool)
target = torch.randn(B, guidance.scattering_dim)

grad_X, grad_E, loss = guidance.compute_guidance(pred_X, pred_E, node_mask, target, return_loss=True)

print("=" * 60)
print("NODE GUIDANCE (MOOD-normalized)")
print("=" * 60)

for scale in [0.1, 1.0, 5.0]:
    guided_X = apply_guidance_to_probs(pred_X, grad_X, scale)
    delta = (guided_X - pred_X).abs().mean().item()
    with torch.no_grad():
        new_moments = guidance.compute_moments(guided_X, pred_E, node_mask)
        new_loss = F.mse_loss(new_moments, target, reduction='sum').item()
    print(f"  scale={scale:5.1f} | delta={delta:.6f} | loss_change={loss - new_loss:+.2f}")

# ── Edge guidance test ──
print()
print("=" * 60)
print("EDGE GUIDANCE (temperature softening + count penalty)")
print("=" * 60)

# Make edges peaked like real model (98% no-bond)
peaked_E = torch.zeros(B, N, N, E)
peaked_E[:, :, :, 0] = 0.98
peaked_E[:, :, :, 1] = 0.01
peaked_E[:, :, :, 2] = 0.005
peaked_E[:, :, :, 3] = 0.003
peaked_E[:, :, :, 4] = 0.002

initial_bonds = compute_expected_bond_count(peaked_E, node_mask)
print(f"  Initial expected bonds: {initial_bonds[0].item():.1f}")
print(f"  Edge dist (0,0): {peaked_E[0,0,0,:].numpy()}")
print()

# Get scattering gradients for edges
_, grad_E_peaked, _ = guidance.compute_guidance(pred_X, peaked_E, node_mask, target, return_loss=True)

# Test different tau values with target of 10 bonds
target_bonds = torch.tensor([10.0] * B)

print(f"  Target bonds: {target_bonds[0].item():.0f}")
print()

for tau in [1.0, 2.0, 3.0, 5.0]:
    for gamma in [0.5, 1.0, 2.0]:
        guided_E, diag = guided_edge_step(
            peaked_E, grad_E_peaked, target_bonds, node_mask,
            tau=tau, gamma=gamma, base_ratio=1.0,
            return_diagnostics=True,
        )
        print(f"  tau={tau:.1f} gamma={gamma:.1f} | "
              f"bonds: {diag['expected_bonds_before']:.1f}->{diag['expected_bonds_after']:.1f} | "
              f"edges_changed: {diag['edges_changed']} | "
              f"max_delta: {diag['max_edge_delta']:.4f}")

# Direction check: does edge guidance reduce scattering loss?
print()
print("=" * 60)
print("DIRECTION CHECK (does edge guidance help scattering?)")
print("=" * 60)

with torch.no_grad():
    orig_moments = guidance.compute_moments(pred_X, peaked_E, node_mask)
    orig_dist = (orig_moments - target).norm(dim=-1).mean().item()

for tau in [2.0, 5.0]:
    guided_E, _ = guided_edge_step(
        peaked_E, grad_E_peaked, target_bonds, node_mask,
        tau=tau, gamma=1.0, base_ratio=1.0,
        return_diagnostics=True,
    )
    with torch.no_grad():
        new_moments = guidance.compute_moments(pred_X, guided_E, node_mask)
        new_dist = (new_moments - target).norm(dim=-1).mean().item()
    status = "OK" if new_dist < orig_dist else "AWAY"
    print(f"  tau={tau:.1f} | L2 dist: {orig_dist:.4f} -> {new_dist:.4f} | {status}")
