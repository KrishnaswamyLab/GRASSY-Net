"""
Sanity tests for MomentMatchingLoss.
Run: python -m moment_diffusion.test_moment_loss
"""
import torch
import torch.nn.functional as F


def test_moment_loss():
    from moment_diffusion.moment_loss import MomentMatchingLoss

    B, N, Xdim, Edim = 4, 10, 7, 5
    num_atom_types = Xdim  # must match

    loss_fn = MomentMatchingLoss(
        num_atom_types=num_atom_types, J=4, num_moments=4,
        lambda_base=0.1, t_weighting=True
    )

    # Create fake predictions (requires grad to test gradient flow)
    pred_X = torch.randn(B, N, Xdim, requires_grad=True)
    pred_E = torch.randn(B, N, N, Edim, requires_grad=True)
    node_mask = torch.ones(B, N, dtype=torch.bool)
    node_mask[:, -2:] = False  # mask last 2 nodes

    # Create fake conditioning (same dim as DenseSoftScattering output)
    scat_dim = loss_fn.soft_scattering.out_shape()
    scattering_cond = torch.randn(B, scat_dim)

    t = torch.randint(0, 500, (B,)).float()

    # Forward
    weighted_loss, raw_loss = loss_fn(pred_X, pred_E, node_mask, scattering_cond, t, 500)

    # Check 1: losses are scalar
    assert weighted_loss.dim() == 0, f"weighted_loss should be scalar, got shape {weighted_loss.shape}"
    assert raw_loss.dim() == 0, f"raw_loss should be scalar, got shape {raw_loss.shape}"
    print(f"1. Loss is scalar: weighted={weighted_loss.item():.6f}, raw={raw_loss.item():.6f}")

    # Check 2: gradients flow to predictions
    weighted_loss.backward()
    assert pred_X.grad is not None, "No gradient on pred_X!"
    assert pred_E.grad is not None, "No gradient on pred_E!"
    assert pred_X.grad.abs().sum() > 0, "pred_X grad is all zeros!"
    assert pred_E.grad.abs().sum() > 0, "pred_E grad is all zeros!"
    print(f"2. Gradients flow: pred_X grad norm={pred_X.grad.norm():.6f}, pred_E grad norm={pred_E.grad.norm():.6f}")

    # Check 3: timestep weighting works
    loss_fn_no_weight = MomentMatchingLoss(
        num_atom_types=num_atom_types, J=4, num_moments=4,
        lambda_base=0.1, t_weighting=False
    )

    # High t (noisy) should have lower weighted loss than low t (clean)
    t_low = torch.zeros(B)    # t=0 -> weight = 1.0
    t_high = torch.full((B,), 499.0)  # t=499 -> weight ~= 0.0

    pred_X2 = torch.randn(B, N, Xdim)
    pred_E2 = torch.randn(B, N, N, Edim)

    loss_low, _ = loss_fn(pred_X2, pred_E2, node_mask, scattering_cond, t_low, 500)
    loss_high, _ = loss_fn(pred_X2, pred_E2, node_mask, scattering_cond, t_high, 500)

    assert loss_low > loss_high, f"Low-t loss ({loss_low.item()}) should be > high-t loss ({loss_high.item()}) due to weighting"
    print(f"3. Timestep weighting: t=0 loss={loss_low.item():.6f} > t=499 loss={loss_high.item():.6f}")

    # Check 4: when predictions match conditioning, loss should be low
    # Use the soft scattering to compute "ground truth" from known inputs
    soft_X = F.softmax(pred_X2, dim=-1)
    soft_E = F.softmax(pred_E2, dim=-1)
    S_true = loss_fn.soft_scattering(soft_X, soft_E, node_mask)

    loss_perfect, raw_perfect = loss_fn_no_weight(pred_X2, pred_E2, node_mask, S_true, t_low, 500)
    assert raw_perfect.item() < 1e-6, f"Loss should be ~0 when prediction matches conditioning, got {raw_perfect.item()}"
    print(f"4. Perfect match loss: {raw_perfect.item():.8f} (should be ~0)")

    print("\nAll tests passed!")


if __name__ == "__main__":
    test_moment_loss()
