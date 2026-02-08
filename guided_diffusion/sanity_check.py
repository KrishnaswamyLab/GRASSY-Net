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

# Test MOOD-normalized guidance at wide range of scales
# (normalization might be too conservative — test bigger values too)
print()
print("=" * 60)
print("MOOD-NORMALIZED GUIDANCE (Fix A + Fix B)")
print("=" * 60)

for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
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

    print(f"  scale={scale:5.1f} | delta={delta:.6f} | eff_ratio={effective_ratio:.4f} | loss_change={loss_reduction:+.2f}")

# Direction check at multiple scales
print()
print("=" * 60)
print("DIRECTION CHECK (does guidance move toward target?)")
print("=" * 60)

with torch.no_grad():
    orig_moments = guidance.compute_moments(pred_X, pred_E, node_mask)
    orig_dist = (orig_moments - target).norm(dim=-1).mean().item()

for scale in [0.1, 1.0, 5.0, 10.0]:
    guided_X = apply_guidance_to_probs(pred_X, grad_X, scale)
    guided_E = apply_guidance_to_probs(pred_E, grad_E, scale)
    with torch.no_grad():
        new_moments = guidance.compute_moments(guided_X, guided_E, node_mask)
        new_dist = (new_moments - target).norm(dim=-1).mean().item()
    status = "OK" if new_dist < orig_dist else "AWAY"
    print(f"  scale={scale:5.1f} | L2 dist: {orig_dist:.4f} -> {new_dist:.4f} | {status}")
