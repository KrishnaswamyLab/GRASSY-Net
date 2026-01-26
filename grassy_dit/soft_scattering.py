"""
Dense Soft Scattering Transform for Differentiable Generation

This module computes scattering moments from soft (probabilistic) graph representations,
enabling gradient flow through the generation process via Gumbel-Softmax.

Unlike the sparse PyG-based ScatteringTransform, this operates on dense batched tensors:
- soft_X: [B, N, num_atom_types] soft atom type probabilities
- soft_E: [B, N, N, num_bond_types] soft edge type probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_dense_moments(x, node_mask, num_moments=4):
    """
    Compute statistical moments per graph from dense batched node features.

    Args:
        x: Node features [B, N, F] where F is feature dimension
        node_mask: [B, N] boolean mask for valid nodes
        num_moments: Number of moments (1=mean, 2=+var, 3=+skew, 4=+kurtosis)

    Returns:
        Moments tensor [B, F * num_moments]
    """
    B, N, F = x.shape
    device = x.device
    dtype = x.dtype
    eps = 1e-8

    # Count valid nodes per graph
    counts = node_mask.float().sum(dim=1, keepdim=True).clamp(min=1)  # [B, 1]

    # Mask invalid nodes
    mask_expanded = node_mask.unsqueeze(-1).float()  # [B, N, 1]
    x_masked = x * mask_expanded  # [B, N, F]

    # Mean
    sum_x = x_masked.sum(dim=1)  # [B, F]
    mean = sum_x / counts  # [B, F]
    moments = [mean]

    if num_moments >= 2:
        # Variance
        mean_expanded = mean.unsqueeze(1)  # [B, 1, F]
        diff = (x - mean_expanded) * mask_expanded  # [B, N, F]
        diff_sq = diff ** 2
        sum_sq = diff_sq.sum(dim=1)  # [B, F]
        variance = sum_sq / counts  # [B, F]
        moments.append(variance)

    if num_moments >= 3:
        # Skewness
        diff_cubed = diff ** 3
        sum_cubed = diff_cubed.sum(dim=1)  # [B, F]
        m3 = sum_cubed / counts
        std_cubed = (variance + eps) ** 1.5
        skew = m3 / std_cubed
        skew = torch.clamp(skew, -1e6, 1e6)
        skew = torch.nan_to_num(skew, nan=0.0)
        moments.append(skew)

    if num_moments >= 4:
        # Kurtosis (Fisher's definition)
        diff_fourth = diff ** 4
        sum_fourth = diff_fourth.sum(dim=1)  # [B, F]
        m4 = sum_fourth / counts
        var_sq = (variance + eps) ** 2
        kurtosis = m4 / var_sq - 3
        kurtosis = torch.clamp(kurtosis, -1e6, 1e6)
        kurtosis = torch.nan_to_num(kurtosis, nan=-3.0)
        moments.append(kurtosis)

    return torch.cat(moments, dim=1)  # [B, F * num_moments]


class DenseSoftScattering(nn.Module):
    """
    Dense Soft Scattering Transform for differentiable graph generation.

    Computes scattering moments from soft atom/edge probabilities, enabling
    gradient flow through Gumbel-Softmax sampling.

    Args:
        num_atom_types: Number of atom type categories
        J: Number of wavelet scales (default: 4)
        num_moments: Number of statistical moments (1-4, default: 4)
    """

    def __init__(self, num_atom_types, J=4, num_moments=4):
        super().__init__()
        self.num_atom_types = num_atom_types
        self.J = J
        self.num_moments = num_moments

        # Wavelet scales: P^1, P^2, P^4, P^8, ... (powers of 2)
        self.scales = [2 ** j for j in range(J)]
        self.max_scale = self.scales[-1]

        # Feng indices for second-order (j' > j pairs)
        self._feng_indices = self._compute_feng_indices()

        # Coefficient counts
        self.num_zeroth = 1
        self.num_first = J
        self.num_second = len(self._feng_indices)
        self.num_total_coeffs = self.num_zeroth + self.num_first + self.num_second

    def _compute_feng_indices(self):
        """Compute valid (j, j') pairs where j' > j."""
        indices = []
        for j in range(self.J):
            for jp in range(j + 1, self.J):
                indices.append(j * self.J + jp)
        return indices

    def _soft_adjacency(self, soft_E, node_mask):
        """
        Build soft adjacency matrix from edge probabilities.

        Args:
            soft_E: [B, N, N, num_bond_types] soft edge predictions
                    Assumes channel 0 is "no bond"
            node_mask: [B, N] valid nodes

        Returns:
            soft_adj: [B, N, N] weighted adjacency (sum of bond probabilities)
        """
        # Sum over bond types, excluding "no bond" (channel 0)
        if soft_E.shape[-1] > 1:
            soft_adj = soft_E[:, :, :, 1:].sum(dim=-1)  # [B, N, N]
        else:
            soft_adj = soft_E.squeeze(-1)

        # Mask invalid edges
        edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)  # [B, N, N]
        soft_adj = soft_adj * edge_mask.float()

        # Zero out diagonal (no self-loops)
        B, N, _ = soft_adj.shape
        diag_mask = ~torch.eye(N, device=soft_adj.device, dtype=torch.bool).unsqueeze(0)
        soft_adj = soft_adj * diag_mask.float()

        return soft_adj

    def _normalize_adjacency(self, adj, node_mask):
        """
        Row-normalize adjacency to get transition matrix P = D^{-1} A.

        Args:
            adj: [B, N, N] adjacency matrix
            node_mask: [B, N] valid nodes

        Returns:
            P: [B, N, N] row-stochastic transition matrix
        """
        # Compute degree
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, N, 1]

        # Row normalize
        P = adj / degree  # [B, N, N]

        return P

    def _lazy_diffuse(self, x, P, node_mask):
        """
        Apply one step of lazy random walk: 0.5 * (x + P @ x)

        Args:
            x: [B, N, F] node features
            P: [B, N, N] transition matrix
            node_mask: [B, N] valid nodes

        Returns:
            diffused: [B, N, F]
        """
        propagated = torch.bmm(P, x)  # [B, N, F]
        diffused = 0.5 * (x + propagated)

        # Mask invalid nodes
        mask = node_mask.unsqueeze(-1).float()  # [B, N, 1]
        return diffused * mask

    def _diffuse_to_scales(self, x, P, node_mask):
        """
        Compute diffusion at powers of 2: P^1, P^2, P^4, P^8, ...

        Returns dict mapping scale -> diffused features
        """
        scale_features = {0: x}  # P^0 = identity
        current = x

        for step in range(1, self.max_scale + 1):
            current = self._lazy_diffuse(current, P, node_mask)
            if step in self.scales:
                scale_features[step] = current

        return scale_features

    def forward(self, soft_X, soft_E, node_mask):
        """
        Compute scattering transform from soft graph representation.

        Args:
            soft_X: [B, N, num_atom_types] soft atom type probabilities
            soft_E: [B, N, N, num_bond_types] soft edge type probabilities
            node_mask: [B, N] boolean mask for valid nodes

        Returns:
            scattering: [B, out_shape()] graph-level scattering moments
        """
        B, N, C = soft_X.shape
        device = soft_X.device

        # Build soft adjacency and transition matrix
        soft_adj = self._soft_adjacency(soft_E, node_mask)  # [B, N, N]
        P = self._normalize_adjacency(soft_adj, node_mask)  # [B, N, N]

        # Use soft atom features as input
        x = soft_X  # [B, N, C]

        # Mask invalid nodes
        mask = node_mask.unsqueeze(-1).float()  # [B, N, 1]
        x = x * mask

        # Compute diffusion at all needed scales
        scale_features = self._diffuse_to_scales(x, P, node_mask)

        # ===== Zeroth-order: low-pass at largest scale =====
        S0 = scale_features[self.max_scale]  # [B, N, C]

        # ===== First-order: wavelet = P^{2^{j-1}} - P^{2^j} =====
        S1_list = []
        for j in range(self.J):
            low_scale = self.scales[j] // 2 if j > 0 else 0
            high_scale = self.scales[j]
            psi_j = scale_features[low_scale] - scale_features[high_scale]
            S1_list.append(torch.abs(psi_j))

        S1 = torch.stack(S1_list, dim=2)  # [B, N, J, C]

        # ===== Second-order: diffuse S1 and apply wavelets again =====
        # Reshape S1 for diffusion: [B, N, J*C]
        U1 = S1.reshape(B, N, self.J * C)

        # Diffuse U1 at all scales
        U1_scales = {0: U1}
        current = U1
        for step in range(1, self.max_scale + 1):
            current = self._lazy_diffuse(current, P, node_mask)
            if step in self.scales:
                U1_scales[step] = current

        # Apply second-order wavelets
        S2_all = []
        for j in range(self.J):
            col_start = j * C
            col_end = (j + 1) * C

            for jp in range(self.J):
                low_scale = self.scales[jp] // 2 if jp > 0 else 0
                high_scale = self.scales[jp]

                psi_jp_Uj = (
                    U1_scales[low_scale][:, :, col_start:col_end]
                    - U1_scales[high_scale][:, :, col_start:col_end]
                )
                S2_all.append(torch.abs(psi_jp_Uj))

        # Stack and select valid pairs: [B, N, J*J, C] -> [B, N, num_second, C]
        S2_stacked = torch.stack(S2_all, dim=2)  # [B, N, J*J, C]
        S2 = S2_stacked[:, :, self._feng_indices, :]  # [B, N, num_second, C]

        # ===== Combine all coefficients =====
        S0_expanded = S0.unsqueeze(2)  # [B, N, 1, C]
        S1_reordered = S1  # [B, N, J, C]
        all_coeffs = torch.cat([S0_expanded, S1_reordered, S2], dim=2)  # [B, N, num_coeffs, C]
        all_coeffs = all_coeffs.reshape(B, N, -1)  # [B, N, num_coeffs * C]

        # ===== Aggregate via moments =====
        scattering = compute_dense_moments(all_coeffs, node_mask, self.num_moments)

        return scattering

    def out_shape(self):
        """Return output feature dimension."""
        return self.num_atom_types * self.num_total_coeffs * self.num_moments


def gumbel_softmax_sample(logits, temperature=1.0, hard=False):
    """
    Sample from Gumbel-Softmax distribution.

    Args:
        logits: [*, num_classes] unnormalized log probabilities
        temperature: Temperature for softmax (lower = more discrete)
        hard: If True, use straight-through estimator for hard samples

    Returns:
        samples: [*, num_classes] soft or hard samples
    """
    return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)


class GumbelSoftmaxSampler:
    """
    Gumbel-Softmax sampler for discrete diffusion.

    Replaces hard multinomial sampling with differentiable soft sampling.
    """

    def __init__(self, temperature=1.0, hard=False):
        """
        Args:
            temperature: Softmax temperature (lower = more discrete)
            hard: Use straight-through estimator for hard samples
        """
        self.temperature = temperature
        self.hard = hard

    def sample(self, prob_X, prob_E, node_mask):
        """
        Sample from probability distributions using Gumbel-Softmax.

        Args:
            prob_X: [B, N, num_atom_types] atom type probabilities
            prob_E: [B, N, N, num_bond_types] edge type probabilities
            node_mask: [B, N] valid nodes mask

        Returns:
            X_s: [B, N, num_atom_types] soft/hard atom samples
            E_s: [B, N, N, num_bond_types] soft/hard edge samples
        """
        B, N, dx = prob_X.shape
        de = prob_E.shape[-1]

        # Convert probabilities to logits (add small epsilon for stability)
        eps = 1e-10
        logits_X = torch.log(prob_X.clamp(min=eps))
        logits_E = torch.log(prob_E.clamp(min=eps))

        # Sample atoms
        X_s = gumbel_softmax_sample(logits_X, self.temperature, self.hard)

        # Sample edges
        # Reshape for sampling: [B, N, N, de] -> [B*N*N, de]
        logits_E_flat = logits_E.reshape(-1, de)
        E_s_flat = gumbel_softmax_sample(logits_E_flat, self.temperature, self.hard)
        E_s = E_s_flat.reshape(B, N, N, de)

        # Make edges symmetric
        E_s = (E_s + E_s.transpose(1, 2)) / 2

        # Mask invalid nodes/edges
        node_mask_X = node_mask.unsqueeze(-1).float()  # [B, N, 1]
        edge_mask = (node_mask.unsqueeze(1) & node_mask.unsqueeze(2)).unsqueeze(-1).float()

        X_s = X_s * node_mask_X
        E_s = E_s * edge_mask

        return X_s, E_s
