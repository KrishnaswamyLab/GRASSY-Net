"""
Graph Scattering Transform for PyTorch Geometric

Implements fixed (non-learnable) graph wavelets via diffusion-based scattering transforms.
Based on:
- Geometric Scattering Networks (Gao et al., 2019)

The scattering transform produces a fixed-size graph-level representation by:
1. Computing diffusion powers P^k on node features
2. Applying wavelet filters (differences of diffusion scales)
3. Taking absolute values (non-linearity)
4. Aggregating via statistical moments (mean, var, skew, kurtosis)
"""

import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.utils.num_nodes import maybe_num_nodes


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, dtype=None):
    """
    Compute row-normalized adjacency: D^{-1} A

    Args:
        edge_index: Edge indices [2, E]
        edge_weight: Optional edge weights [E]
        num_nodes: Number of nodes (inferred if None)
        dtype: Data type for edge weights

    Returns:
        edge_index: Edge indices
        edge_weight: Normalized edge weights
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1),),
            dtype=dtype,
            device=edge_index.device,
        )

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)

    return edge_index, deg_inv[row] * edge_weight


class Diffusion(MessagePassing):
    """
    Lazy random walk diffusion operator (fixed, non-learnable).

    Implements: P x = 0.5 * (x + A D^{-1} x)
    """

    def __init__(self):
        super().__init__(aggr="add", node_dim=0)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Apply one step of lazy diffusion.

        Args:
            x: Node features [N, C] or [N, C, ...]
            edge_index: Edge indices [2, E]
            edge_weight: Optional edge weights [E]

        Returns:
            Diffused features with same shape as input
        """
        edge_index, edge_weight = gcn_norm(
            edge_index, edge_weight, x.size(0), dtype=x.dtype
        )
        propagated = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return 0.5 * (x + propagated)

    def message(self, x_j, edge_weight):
        if x_j.dim() == 2:
            return edge_weight.view(-1, 1) * x_j
        else:
            shape = [-1] + [1] * (x_j.dim() - 1)
            return edge_weight.view(*shape) * x_j


def compute_moments(x, batch, num_moments=4):
    """
    Compute statistical moments per graph.

    Args:
        x: Node features [N, F] where F is feature dimension
        batch: Batch indices [N] assigning nodes to graphs
        num_moments: Number of moments (1=mean, 2=+var, 3=+skew, 4=+kurtosis)

    Returns:
        Moments tensor [B, F * num_moments] where B is number of graphs
    """
    device = x.device
    dtype = x.dtype
    num_graphs = batch.max().item() + 1

    # Count nodes per graph
    ones = torch.ones(x.size(0), device=device, dtype=dtype)
    counts = scatter_add(ones, batch, dim=0, dim_size=num_graphs)
    counts = counts.clamp(min=1)

    # Mean
    sum_x = scatter_add(x, batch, dim=0, dim_size=num_graphs)
    mean = sum_x / counts.unsqueeze(1)
    moments = [mean]

    if num_moments >= 2:
        # Variance
        mean_expanded = mean[batch]
        diff = x - mean_expanded
        diff_sq = diff ** 2
        sum_sq = scatter_add(diff_sq, batch, dim=0, dim_size=num_graphs)
        variance = sum_sq / counts.unsqueeze(1)
        moments.append(variance)

    if num_moments >= 3:
        # Skewness
        eps = 1e-8
        diff_cubed = diff ** 3
        sum_cubed = scatter_add(diff_cubed, batch, dim=0, dim_size=num_graphs)
        m3 = sum_cubed / counts.unsqueeze(1)
        std_cubed = (variance + eps) ** 1.5
        skew = m3 / std_cubed
        skew = torch.clamp(skew, -1e6, 1e6)
        skew = torch.nan_to_num(skew, nan=0.0)
        moments.append(skew)

    if num_moments >= 4:
        # Kurtosis (Fisher's definition)
        diff_fourth = diff ** 4
        sum_fourth = scatter_add(diff_fourth, batch, dim=0, dim_size=num_graphs)
        m4 = sum_fourth / counts.unsqueeze(1)
        var_sq = (variance + eps) ** 2
        kurtosis = m4 / var_sq - 3
        kurtosis = torch.clamp(kurtosis, -1e6, 1e6)
        kurtosis = torch.nan_to_num(kurtosis, nan=-3.0)
        moments.append(kurtosis)

    return torch.cat(moments, dim=1)


class GraphScatteringTransform(nn.Module):
    """
    Fixed (non-learnable) Graph Scattering Transform.

    Computes a fixed-size representation for each graph via:
    1. Zeroth-order: low-pass filtered features
    2. First-order: |ψ_j * x| wavelet responses at J scales
    3. Second-order: |ψ_{j'} * |ψ_j * x|| for j' > j
    4. Statistical moment aggregation (mean, var, skew, kurtosis)

    Output dimension: in_channels * (1 + J + J*(J-1)/2) * num_moments

    Args:
        in_channels: Number of input node features
        J: Number of wavelet scales (default: 4)
        num_moments: Number of statistical moments (1-4, default: 4)
    """

    def __init__(self, in_channels, J=4, num_moments=4):
        super().__init__()

        self.in_channels = in_channels
        self.J = J
        self.num_moments = num_moments

        # Fixed diffusion operators
        self.diffuse = Diffusion()

        # Wavelet scales: P^1, P^2, P^4, P^8, ... (powers of 2)
        self.scales = [2 ** j for j in range(J)]
        self.max_scale = self.scales[-1]

        # Feng indices for second-order
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

    def _diffuse_to_scales(self, x, edge_index):
        """
        Compute diffusion at powers of 2: P^1, P^2, P^4, P^8, ...

        Returns dict mapping scale -> diffused features
        """
        scale_features = {0: x}  # P^0 = identity
        current = x

        for step in range(1, self.max_scale + 1):
            current = self.diffuse(current, edge_index)
            if step in self.scales:
                scale_features[step] = current

        return scale_features

    def forward(self, data):
        """
        Compute scattering transform for a batch of graphs.

        Args:
            data: PyG Data or Batch object with x, edge_index, batch

        Returns:
            scattering: Graph-level features [B, out_shape()]
        """
        x = data.x  # [N, C] -> N: number of nodes C: number of channels (atom types one-hot)
        edge_index = data.edge_index
        batch = getattr(data, 'batch', None)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        N, C = x.shape # [N, C] -> N: number of nodes C: number of channels (atom types one-hot)

        # Compute diffusion at all needed scales
        scale_features = self._diffuse_to_scales(x, edge_index)

        # ===== Zeroth-order: low-pass at largest scale =====
        S0 = scale_features[self.max_scale]  # [N, C]

        # ===== First-order: wavelet = P^{2^{j-1}} - P^{2^j} =====
        S1_list = []
        for j in range(self.J):
            low_scale = self.scales[j] // 2 if j > 0 else 0
            high_scale = self.scales[j]
            psi_j = scale_features[low_scale] - scale_features[high_scale]
            S1_list.append(torch.abs(psi_j))

        S1 = torch.stack(S1_list, dim=1)  # [N, J, C]

        # ===== Second-order: diffuse S1 and apply wavelets again =====
        # Reshape S1 for diffusion: [N, J*C]
        U1 = S1.reshape(N, self.J * C)

        # Diffuse U1 at all scales
        U1_scales = {0: U1}
        current = U1
        for step in range(1, self.max_scale + 1):
            current = self.diffuse(current, edge_index)
            if step in self.scales:
                U1_scales[step] = current

        # Apply second-order wavelets
        S2_all = []
        for j in range(self.J):
            # Extract j-th first-order channel
            col_start = j * C
            col_end = (j + 1) * C

            for jp in range(self.J):
                low_scale = self.scales[jp] // 2 if jp > 0 else 0
                high_scale = self.scales[jp]

                psi_jp_Uj = (
                    U1_scales[low_scale][:, col_start:col_end]
                    - U1_scales[high_scale][:, col_start:col_end]
                )
                S2_all.append(torch.abs(psi_jp_Uj))

        # Stack and select valid pairs
        S2_stacked = torch.stack(S2_all, dim=1)  # [N, J*J, C]
        S2 = S2_stacked[:, self._feng_indices, :]  # [N, num_second, C]

        # ===== Combine all coefficients =====
        S0_expanded = S0.unsqueeze(1)  # [N, 1, C]
        all_coeffs = torch.cat([S0_expanded, S1, S2], dim=1)  # [N, num_coeffs, C]
        all_coeffs = all_coeffs.reshape(N, -1)  # [N, num_coeffs * C]

        # ===== Aggregate via moments =====
        scattering = compute_moments(all_coeffs, batch, self.num_moments)

        return scattering

    def out_shape(self):
        """Return output feature dimension."""
        return self.in_channels * self.num_total_coeffs * self.num_moments
