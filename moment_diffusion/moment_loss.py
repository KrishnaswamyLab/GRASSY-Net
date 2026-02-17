"""
Single-step moment-matching loss for scattering-conditioned diffusion.

Computes differentiable scattering on the model's x0-prediction (soft probabilities)
and penalizes mismatch with the conditioning scattering moments.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from moment_diffusion.soft_scattering import DenseSoftScattering


class MomentMatchingLoss(nn.Module):
    """
    Auxiliary loss: MSE between predicted molecule's scattering moments
    and the conditioning scattering moments.

    Args:
        num_atom_types: Number of atom type categories (must match DenseSoftScattering)
        J: Wavelet scales (default 4)
        num_moments: Statistical moments (default 4)
        lambda_base: Base loss weight before timestep scaling
        t_weighting: If True, scale loss by (1 - t/T) so high-noise steps contribute less
    """

    def __init__(self, num_atom_types, J=4, num_moments=4,
                 lambda_base=0.1, t_weighting=True):
        super().__init__()
        self.soft_scattering = DenseSoftScattering(
            num_atom_types=num_atom_types, J=J, num_moments=num_moments
        )
        self.lambda_base = lambda_base
        self.t_weighting = t_weighting

    def forward(self, pred_X_logits, pred_E_logits, node_mask,
                scattering_cond, t, timesteps):
        """
        Args:
            pred_X_logits: [B, N, Xdim] -- model's atom prediction (RAW LOGITS, not softmax)
            pred_E_logits: [B, N, N, Edim] -- model's edge prediction (RAW LOGITS)
            node_mask: [B, N] -- valid nodes
            scattering_cond: [B, scat_dim] -- conditioning moments (ground truth)
            t: [B] or [B, 1] -- diffusion timestep for this batch
            timesteps: int -- total diffusion timesteps T (e.g. 500)

        Returns:
            weighted_loss: scalar -- lambda(t) * MSE(S_pred, S_cond)
            loss_raw: scalar -- unweighted MSE (for logging)
        """
        # Convert logits to soft probabilities
        soft_X = F.softmax(pred_X_logits, dim=-1)  # [B, N, Xdim]
        soft_E = F.softmax(pred_E_logits, dim=-1)  # [B, N, N, Edim]

        # Pad atom types if model outputs fewer than scattering expects
        # (e.g. QM9: model has 4 active atom types, scattering computed with 5)
        expected_atoms = self.soft_scattering.num_atom_types
        if soft_X.shape[-1] < expected_atoms:
            pad_size = expected_atoms - soft_X.shape[-1]
            soft_X = F.pad(soft_X, (0, pad_size), value=0.0)

        # Compute scattering of predicted molecule
        S_pred = self.soft_scattering(soft_X, soft_E, node_mask)  # [B, scat_dim]

        # MSE loss
        loss_raw = F.mse_loss(S_pred, scattering_cond)

        # Timestep weighting: lambda(t) = (1 - t/T) * lambda_base
        if self.t_weighting:
            t_flat = t.view(-1).float()  # [B]
            weight = (1.0 - t_flat / timesteps).mean()  # average over batch
            weighted_loss = self.lambda_base * weight * loss_raw
        else:
            weighted_loss = self.lambda_base * loss_raw

        return weighted_loss, loss_raw
