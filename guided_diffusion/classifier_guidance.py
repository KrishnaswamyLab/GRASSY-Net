"""
Classifier Guidance for Graph Diffusion Models.

Computes gradients from a trained moment classifier to guide molecule generation
toward target scattering moments. Unlike direct scattering guidance (which
computes scattering on predicted clean graphs), this uses a learned classifier
that predicts moments from noisy graphs.

Key difference:
- Direct guidance: operates on predicted clean graph (pred_X, pred_E)
- Classifier guidance: operates on current noisy state (X_t, E_t)
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from guided_diffusion.moment_classifier import MomentClassifier, load_classifier


class ClassifierGuidance:
    """
    Computes gradients using a trained moment classifier to guide generation.
    
    At each diffusion timestep, this class:
    1. Takes the current noisy graph state (X_t, E_t)
    2. Predicts scattering moments using the trained classifier
    3. Computes MSE loss to target moments
    4. Returns gradients to shift the generation toward target
    
    Args:
        classifier: Trained MomentClassifier model, or path to checkpoint
        device: Device to run computations on
    """
    
    def __init__(
        self,
        classifier: MomentClassifier | str,
        device: str = 'cuda',
    ):
        self.device = device
        
        # Load classifier if path provided
        if isinstance(classifier, str):
            self.classifier = load_classifier(classifier, device)
        else:
            self.classifier = classifier.to(device)
        
        self.classifier.eval()
        
        # Store dimensions
        self.moment_dim = self.classifier.moment_dim
        self.max_n_nodes = self.classifier.max_n_nodes
        self.Xdim = self.classifier.Xdim
        self.Edim = self.classifier.Edim
    
    def compute_guidance(
        self,
        X_t: torch.Tensor,
        E_t: torch.Tensor,
        t: torch.Tensor,
        node_mask: torch.Tensor,
        target_moments: torch.Tensor,
        return_loss: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Compute gradients w.r.t. noisy graph that push toward target moments.
        
        Args:
            X_t: [B, N, Xdim] noisy atom features
            E_t: [B, N, N, Edim] noisy edge features
            t: [B] or [B, 1] timestep values
            node_mask: [B, N] boolean mask for valid nodes
            target_moments: [B, moment_dim] target scattering moments
            return_loss: If True, also return the MSE loss value
        
        Returns:
            grad_X: [B, N, Xdim] gradient to add to X predictions
            grad_E: [B, N, N, Edim] gradient to add to E predictions
            loss: (optional) scalar MSE loss if return_loss=True
        """
        # Detach from computation graph and enable gradients
        X_t = X_t.detach().clone().requires_grad_(True)
        E_t = E_t.detach().clone().requires_grad_(True)
        
        # Forward pass through classifier
        predicted_moments = self.classifier(X_t, E_t, t, node_mask)
        
        # MSE loss to target moments
        loss = F.mse_loss(predicted_moments, target_moments, reduction='sum')
        
        # Backpropagate to get gradients
        loss.backward()
        
        # Return NEGATIVE gradients (we want to minimize loss)
        grad_X = -X_t.grad
        grad_E = -E_t.grad
        
        if return_loss:
            return grad_X, grad_E, loss.item()
        return grad_X, grad_E
    
    @torch.no_grad()
    def predict_moments(
        self,
        X_t: torch.Tensor,
        E_t: torch.Tensor,
        t: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict scattering moments from noisy graph (no gradient).
        
        Useful for monitoring/debugging during generation.
        
        Args:
            X_t: [B, N, Xdim] noisy atom features
            E_t: [B, N, N, Edim] noisy edge features
            t: [B] or [B, 1] timestep values
            node_mask: [B, N] boolean mask for valid nodes
        
        Returns:
            [B, moment_dim] predicted scattering moments
        """
        return self.classifier(X_t, E_t, t, node_mask)


def apply_classifier_guidance_to_probs(
    prob: torch.Tensor,
    grad: torch.Tensor,
    scale: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Apply classifier guidance gradient to probability distribution.
    
    This function applies the gradient in probability space and re-normalizes.
    The gradient should be the negative gradient of the loss (i.e., direction
    that reduces the loss / improves moment matching).
    
    Args:
        prob: [..., K] probability distribution (sums to 1 over last dim)
        grad: [..., K] gradient to apply (negative of loss gradient)
        scale: Guidance scale (higher = stronger guidance)
        eps: Small value for numerical stability
    
    Returns:
        guided_prob: [..., K] guided probability distribution
    """
    # Apply gradient in probability space
    guided_prob = prob + scale * grad
    
    # Ensure non-negative
    guided_prob = guided_prob.clamp(min=eps)
    
    # Re-normalize to sum to 1
    guided_prob = guided_prob / guided_prob.sum(dim=-1, keepdim=True)
    
    return guided_prob


def apply_classifier_guidance_in_logit_space(
    logits: torch.Tensor,
    grad: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """
    Apply classifier guidance gradient in logit space.
    
    Alternative to probability space guidance. May be more stable for
    large gradients since it doesn't require clamping.
    
    Args:
        logits: [..., K] unnormalized log probabilities
        grad: [..., K] gradient to apply (negative of loss gradient)
        scale: Guidance scale
    
    Returns:
        guided_logits: [..., K] guided logits
    """
    # Apply gradient directly to logits
    guided_logits = logits + scale * grad
    
    return guided_logits


class HybridGuidance:
    """
    Combines classifier guidance with direct scattering guidance.
    
    This allows using both approaches together, potentially getting benefits
    from each:
    - Classifier guidance: operates on noisy state, learned representation
    - Direct guidance: operates on predicted clean state, exact scattering
    
    Args:
        classifier_guidance: ClassifierGuidance instance
        direct_guidance: ScatteringMomentGuidance instance (from guidance.py)
        classifier_weight: Weight for classifier guidance (0 to 1)
        direct_weight: Weight for direct guidance (0 to 1)
    """
    
    def __init__(
        self,
        classifier_guidance: ClassifierGuidance,
        direct_guidance,  # ScatteringMomentGuidance from guidance.py
        classifier_weight: float = 0.5,
        direct_weight: float = 0.5,
    ):
        self.classifier_guidance = classifier_guidance
        self.direct_guidance = direct_guidance
        self.classifier_weight = classifier_weight
        self.direct_weight = direct_weight
    
    def compute_hybrid_guidance(
        self,
        X_t: torch.Tensor,
        E_t: torch.Tensor,
        pred_X: torch.Tensor,
        pred_E: torch.Tensor,
        t: torch.Tensor,
        node_mask: torch.Tensor,
        target_moments: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute both classifier and direct guidance.
        
        Args:
            X_t, E_t: Current noisy state (for classifier guidance)
            pred_X, pred_E: Predicted clean state (for direct guidance)
            t: Timestep
            node_mask: Valid node mask
            target_moments: Target scattering moments
        
        Returns:
            classifier_grad_X, classifier_grad_E: Gradients for noisy state
            direct_grad_X, direct_grad_E: Gradients for predicted state
        """
        # Classifier guidance (operates on noisy state)
        clf_grad_X, clf_grad_E = self.classifier_guidance.compute_guidance(
            X_t, E_t, t, node_mask, target_moments
        )
        
        # Direct scattering guidance (operates on predicted clean state)
        dir_grad_X, dir_grad_E = self.direct_guidance.compute_guidance(
            pred_X, pred_E, node_mask, target_moments
        )
        
        # Weight the gradients
        clf_grad_X = clf_grad_X * self.classifier_weight
        clf_grad_E = clf_grad_E * self.classifier_weight
        dir_grad_X = dir_grad_X * self.direct_weight
        dir_grad_E = dir_grad_E * self.direct_weight
        
        return clf_grad_X, clf_grad_E, dir_grad_X, dir_grad_E
