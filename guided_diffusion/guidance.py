"""
Scattering Moment Guidance for Graph Diffusion Models.

Computes gradients that steer molecule generation toward target scattering moments.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from .soft_scattering import DenseSoftScattering


class ScatteringMomentGuidance:
    """
    Computes gradients steering generation toward target scattering moments.
    
    At each diffusion timestep, this class:
    1. Takes soft (probabilistic) graph predictions
    2. Computes scattering moments via DenseSoftScattering
    3. Computes MSE loss to target moments
    4. Returns gradients to shift predictions toward target
    
    Args:
        num_atom_types: Number of atom type categories (e.g., 10 for ZINC)
        J: Number of wavelet scales (default: 4)
        num_moments: Number of statistical moments per coefficient (default: 4)
        device: Device to run computations on
    """
    
    def __init__(
        self,
        num_atom_types: int,
        J: int = 4,
        num_moments: int = 4,
        device: str = 'cuda'
    ):
        self.device = device
        self.scattering = DenseSoftScattering(
            num_atom_types=num_atom_types,
            J=J,
            num_moments=num_moments
        ).to(device)
        
        # Store dimensions for validation
        self.num_atom_types = num_atom_types
        self.J = J
        self.num_moments = num_moments
        self.scattering_dim = self.scattering.out_shape()
    
    def compute_guidance(
        self,
        pred_X: torch.Tensor,
        pred_E: torch.Tensor,
        node_mask: torch.Tensor,
        target_moments: torch.Tensor,
        return_loss: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gradients that point toward target scattering moments.
        
        Args:
            pred_X: [B, N, V] predicted clean node type probabilities
            pred_E: [B, N, N, E] predicted clean edge type probabilities
            node_mask: [B, N] boolean mask for valid nodes
            target_moments: [B, D] target scattering moments
            return_loss: If True, also return the MSE loss value
        
        Returns:
            grad_X: [B, N, V] gradient to ADD to node probabilities/logits
            grad_E: [B, N, N, E] gradient to ADD to edge probabilities/logits
            loss: (optional) scalar MSE loss if return_loss=True
        """
        # Detach from computation graph and enable gradients for guidance
        pred_X = pred_X.detach().clone().requires_grad_(True)
        pred_E = pred_E.detach().clone().requires_grad_(True)
        
        # Forward pass through differentiable scattering
        current_moments = self.scattering(pred_X, pred_E, node_mask)
        
        # MSE loss to target moments
        loss = F.mse_loss(current_moments, target_moments, reduction='sum')
        
        # Backpropagate to get gradients
        loss.backward()
        
        # Return NEGATIVE gradients (we want to minimize loss, so move opposite to gradient)
        grad_X = -pred_X.grad
        grad_E = -pred_E.grad
        
        if return_loss:
            return grad_X, grad_E, loss.item()
        return grad_X, grad_E
    
    def compute_moments(
        self,
        pred_X: torch.Tensor,
        pred_E: torch.Tensor,
        node_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scattering moments from soft graph predictions.
        
        Args:
            pred_X: [B, N, V] predicted node type probabilities
            pred_E: [B, N, N, E] predicted edge type probabilities
            node_mask: [B, N] boolean mask for valid nodes
        
        Returns:
            moments: [B, D] scattering moments
        """
        with torch.no_grad():
            return self.scattering(pred_X, pred_E, node_mask)


def apply_guidance_to_probs(
    prob: torch.Tensor,
    grad: torch.Tensor,
    scale: float,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Apply guidance gradient to probability distribution.
    
    Works directly in probability space for correct gradient descent.
    
    Args:
        prob: [..., K] probability distribution (sums to 1 over last dim)
        grad: [..., K] gradient to apply (should be -∂loss/∂prob for minimization)
        scale: Guidance scale (higher = stronger guidance)
        eps: Small value for numerical stability
    
    Returns:
        guided_prob: [..., K] guided probability distribution
    """
    # Apply gradient directly in probability space
    guided_prob = prob + scale * grad
    
    # Ensure non-negative
    guided_prob = guided_prob.clamp(min=eps)
    
    # Re-normalize to sum to 1
    guided_prob = guided_prob / guided_prob.sum(dim=-1, keepdim=True)
    
    return guided_prob
