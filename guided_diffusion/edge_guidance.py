"""
Edge Guidance via Temperature Softening with Edge Count Penalty.

The edge posterior in discrete graph diffusion is ~98% peaked on "no bond",
making it impervious to standard gradient guidance. This module combines:

1. Temperature softening: flatten edge distributions so gradients can move them
2. Scattering gradient: push specific edges toward bond types that improve moment matching
3. Edge count penalty: push total expected bonds toward a target, counteracting
   indiscriminate bond creation from softening

The edge count penalty acts as an automatic threshold: the scattering gradient
must overcome the penalty to flip any given edge.
"""

import torch
import torch.nn.functional as F


def compute_expected_bond_count(prob_E: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute expected number of bonds from edge probability distribution.

    Args:
        prob_E: [B, N, N, E] edge type probabilities (index 0 = no bond)
        node_mask: [B, N] boolean mask for valid nodes

    Returns:
        expected_bonds: [B] expected bond count per molecule
    """
    # P(any bond) = 1 - P(no bond)
    p_bond = 1.0 - prob_E[:, :, :, 0]  # [B, N, N]

    # Mask out invalid nodes
    edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  # [B, N, N]
    p_bond = p_bond * edge_mask.float()

    # Upper triangle only (undirected graph — avoid double counting)
    upper_mask = torch.triu(torch.ones_like(p_bond[0]), diagonal=1).unsqueeze(0)
    p_bond = p_bond * upper_mask

    return p_bond.sum(dim=(-1, -2))  # [B]


def compute_edge_count_gradient(
    prob_E: torch.Tensor,
    target_edge_count: torch.Tensor,
    node_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute gradient that pushes expected bond count toward target.

    When expected bonds > target: push no-bond up, bond categories down.
    When expected bonds < target: push no-bond down, bond categories up.

    Args:
        prob_E: [B, N, N, E] edge type probabilities
        target_edge_count: [B] target number of bonds
        node_mask: [B, N] boolean mask

    Returns:
        grad_E_count: [B, N, N, E] gradient for edge count penalty
    """
    expected_bonds = compute_expected_bond_count(prob_E, node_mask)
    bond_excess = expected_bonds - target_edge_count  # positive = too many bonds

    # Build gradient: for each edge position
    # d/d(prob_E[:,:,:,0]) penalty = +bond_excess (push no-bond up when excess)
    # d/d(prob_E[:,:,:,k]) penalty = -bond_excess for k>0 (push bonds down when excess)
    # We negate because compute_guidance returns -grad (minimization direction)
    grad_E_count = torch.zeros_like(prob_E)
    excess = bond_excess[:, None, None, None]  # [B, 1, 1, 1]

    # No-bond channel: negative gradient pushes prob up (reduces bonds)
    grad_E_count[:, :, :, 0] = -excess.squeeze(-1)
    # Bond channels: positive gradient pushes prob up (adds bonds)
    for k in range(1, prob_E.shape[-1]):
        grad_E_count[:, :, :, k] = excess.squeeze(-1)

    return grad_E_count, bond_excess


def guided_edge_step(
    posterior_E: torch.Tensor,
    grad_E_scat: torch.Tensor,
    target_edge_count: torch.Tensor,
    node_mask: torch.Tensor,
    tau: float = 2.0,
    gamma: float = 1.0,
    base_ratio: float = 1.0,
    eps: float = 1e-10,
    return_diagnostics: bool = False,
) -> torch.Tensor:
    """
    Apply edge guidance with temperature softening and edge count penalty.

    Args:
        posterior_E: [B, N, N, E] edge posterior probabilities
        grad_E_scat: [B, N, N, E] scattering loss gradients (already negated)
        target_edge_count: [B] target number of bonds
        node_mask: [B, N] boolean mask for valid nodes
        tau: Temperature for softening (>1 flattens, try 2.0-5.0)
        gamma: Weight of edge count penalty relative to scattering gradient
        base_ratio: MOOD-style guidance scale
        eps: Numerical stability
        return_diagnostics: If True, return diagnostic dict

    Returns:
        guided_E: [B, N, N, E] guided edge probabilities
        diagnostics: (optional) dict with diagnostic info
    """
    # Step 1: Compute edge count penalty gradient
    grad_E_count, bond_excess = compute_edge_count_gradient(
        posterior_E, target_edge_count, node_mask
    )

    # Normalize count gradient to match scale of scattering gradient
    count_norm = grad_E_count.abs().mean() + eps
    scat_norm = grad_E_scat.abs().mean() + eps
    grad_E_count = grad_E_count * (scat_norm / count_norm)

    # Step 2: Combine gradients
    combined_grad = grad_E_scat + gamma * grad_E_count

    # Step 3: Soften, guide, sharpen in logit space
    logits_E = torch.log(posterior_E + eps)
    softened_logits = logits_E / tau  # flatten distribution

    # MOOD-style normalization for combined gradient
    pred_norm = posterior_E.abs().mean()
    grad_norm = combined_grad.abs().mean() + eps
    normalized_scale = base_ratio * pred_norm / grad_norm

    # Apply guidance to softened logits
    guided_logits = softened_logits + normalized_scale * combined_grad

    # Sharpen back
    resharpened_logits = guided_logits * tau

    # Back to probabilities
    guided_E = F.softmax(resharpened_logits, dim=-1)

    if return_diagnostics:
        p_bond_before = compute_expected_bond_count(posterior_E, node_mask)
        p_bond_after = compute_expected_bond_count(guided_E, node_mask)
        delta_E = guided_E - posterior_E

        diagnostics = {
            'target_edge_count': target_edge_count[0].item(),
            'expected_bonds_before': p_bond_before[0].item(),
            'expected_bonds_after': p_bond_after[0].item(),
            'bond_excess': bond_excess[0].item(),
            'scat_grad_norm': scat_norm.item(),
            'count_grad_norm': count_norm.item(),
            'edges_changed': (delta_E.abs() > 0.01).sum().item(),
            'max_edge_delta': delta_E.abs().max().item(),
            'tau': tau,
            'gamma': gamma,
            'base_ratio': base_ratio,
        }
        return guided_E, diagnostics

    return guided_E
