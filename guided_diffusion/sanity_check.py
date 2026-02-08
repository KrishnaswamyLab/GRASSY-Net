"""Quick local sanity check for guidance math. No checkpoint needed."""
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, '.')

from guided_diffusion.guidance import ScatteringMomentGuidance, apply_guidance_to_probs

device = 'cpu'
B, N, V, E = 2, 10, 10, 5
guidance = ScatteringMomentGuidance(num_atom_types=V, J=4, num_moments=4, device=device)

pred_X = F.softmax(torch.randn(B, N, V), dim=-1)
pred_E = F.softmax(torch.randn(B, N, N, E), dim=-1)
node_mask = torch.ones(B, N, dtype=torch.bool)
target = torch.randn(B, guidance.scattering_dim)

grad_X, grad_E, loss = guidance.compute_guidance(pred_X, pred_E, node_mask, target, return_loss=True)

# Raw gradient stats
raw_ratio_X = (grad_X.abs().mean()) / (pred_X.abs().mean() + 1e-10)
raw_ratio_E = (grad_E.abs().mean()) / (pred_E.abs().mean() + 1e-10)

print("=" * 60)
print("RAW GRADIENT STATS (before normalization)")
print("=" * 60)
print(f"  loss:       {loss:.4f}")
print(f"  raw_ratio_X: {raw_ratio_X:.2f}  (grad magnitude / pred magnitude)")
print(f"  raw_ratio_E: {raw_ratio_E:.2f}")
print(f"  grad_X range: [{grad_X.min():.4f}, {grad_X.max():.4f}]")
print(f"  pred_X range: [{pred_X.min():.4f}, {pred_X.max():.4f}]")

# Test MOOD-normalized guidance at different scales
print()
print("=" * 60)
print("MOOD-NORMALIZED GUIDANCE (Fix A)")
print("=" * 60)

for scale in [0.01, 0.05, 0.1, 0.2, 0.5]:
    guided_X = apply_guidance_to_probs(pred_X, grad_X, scale)
    delta = (guided_X - pred_X).abs().mean().item()

    # Effective ratio after normalization
    pred_norm = pred_X.abs().mean().item()
    effective_ratio = delta / (pred_norm + 1e-10)

    # Check guidance direction: does it reduce loss?
    with torch.no_grad():
        new_moments = guidance.compute_moments(guided_X, pred_E, node_mask)
        new_loss = F.mse_loss(new_moments, target, reduction='sum').item()
    loss_reduction = loss - new_loss

    print(f"  scale={scale:.2f} | delta={delta:.6f} | effective_ratio={effective_ratio:.4f} | loss_change={loss_reduction:+.2f}")

# Verify the best scale actually reduces loss
print()
guided_X_01 = apply_guidance_to_probs(pred_X, grad_X, 0.1)
with torch.no_grad():
    orig_moments = guidance.compute_moments(pred_X, pred_E, node_mask)
    new_moments = guidance.compute_moments(guided_X_01, pred_E, node_mask)
    orig_dist = (orig_moments - target).norm(dim=-1).mean().item()
    new_dist = (new_moments - target).norm(dim=-1).mean().item()

print("=" * 60)
print("DIRECTION CHECK (scale=0.1)")
print("=" * 60)
print(f"  L2 dist before guidance: {orig_dist:.4f}")
print(f"  L2 dist after guidance:  {new_dist:.4f}")
if new_dist < orig_dist:
    print("  OK: Guidance moves predictions toward target")
else:
    print("  WARNING: Guidance moves AWAY from target — sign issue?")
