# scatter.py
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear
from torch_scatter import scatter_add
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import to_dense_batch


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, add_self_loops=False, dtype=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones(
            (edge_index.size(1),),
            dtype=dtype,
            device=edge_index.device,
        )

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 1, num_nodes
        )
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv = deg.pow(-1)
    deg_inv.masked_fill_(deg_inv == float("inf"), 0)

    return edge_index, deg_inv[row] * edge_weight


class Diffuse(MessagePassing):
    """
    Implements lazy random walk:
        P x = 0.5 (x + A D^{-1} x)
    """

    def __init__(self, channels):
        super().__init__(aggr="add", node_dim=0)
        self.channels = channels

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, edge_weight = gcn_norm(
            edge_index, edge_weight, x.size(0), dtype=x.dtype
        )
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)
        return 0.5 * (x + out)

    def message(self, x_j, edge_weight):
        return edge_weight.view(-1, 1) * x_j


def feng_filters():
    results = [4]
    for i in range(2, 4):
        for j in range(0, i):
            results.append(4 * i + j)
    return results


class Scatter_layer(nn.Module):
    """
    Implements:
        h^(0) = x
        h^(l) = MLP_l( ψ_l h^(l-1) )
        ψ_l = sum_k F[l,k] P^k − sum_k F[l−1,k] P^k
    """

    def __init__(self, in_channels, max_graph_size, trainable_f=False):
        super().__init__()

        self.in_channels = in_channels
        self.max_graph_size = max_graph_size

        # Diffusion operator P
        self.diffuse = Diffuse(in_channels)

        # IMPORTANT:
        # Do NOT force .to(device) here. Lightning / .to(...) on the parent model will move this Parameter.
        self.wavelet_constructor = nn.Parameter(
            torch.tensor(
                [
                    [0, -1.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1],
                ],
                dtype=torch.float32,
            ),
            requires_grad=trainable_f,
        )

        self.lin_in = nn.Linear(in_channels, in_channels, bias=False)
        self.L = self.wavelet_constructor.shape[0]
        self.K = self.wavelet_constructor.shape[1] - 1

        # One MLP per wavelet layer
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_channels, in_channels),
                    nn.ReLU(),
                    nn.Linear(in_channels, in_channels),
                )
                for _ in range(self.L)
            ]
        )

    def forward(self, data, return_f_matrix=False):
        x = data.x
        edge_index = data.edge_index

        # Ensure edge_index is on same device as x (common source of cpu/cuda mixups)
        if edge_index.device != x.device:
            edge_index = edge_index.to(x.device)

        N, C = x.shape

        # h^(0)
        h = self.lin_in(x)
        layer_outputs = []

        for l in range(self.L):
            # Compute diffusion powers P^k h
            diffusions = torch.empty(self.K + 1, N, C, device=x.device, dtype=h.dtype)
            diffusions[0] = h
            cur = h
            for k in range(1, self.K + 1):
                cur = self.diffuse(cur, edge_index)
                diffusions[k] = cur

            # ψ_l weights = (F[l] − F[l−1]) or F[0]
            if l == 0:
                weights = self.wavelet_constructor[l]
            else:
                weights = self.wavelet_constructor[l] - self.wavelet_constructor[l - 1]

            # Ensure weights matches diffusions device + dtype
            weights = weights.to(device=diffusions.device, dtype=diffusions.dtype)

            psi_h = torch.einsum("k,knd->nd", weights, diffusions)
            psi_h = torch.abs(psi_h)

            # Update
            h = self.mlps[l](psi_h)
            layer_outputs.append(h)

        # Stack scattering coefficients
        x = torch.stack(layer_outputs, dim=2)  # [N, C, L]
        x = x.permute(0, 2, 1)                 # [N, L, C]

        # to dense [B, max_nodes, L, C]
        x_dense = to_dense_batch(x, data.batch)[0]  # [B, max_nodes, L, C]

        # flatten node dimension into "sequence" dimension expected downstream
        x = x_dense.reshape(data.num_graphs, -1, self.out_shape())

        if return_f_matrix:
            return x, self.wavelet_constructor
        
        return x

    def out_shape(self):
        return self.L * self.in_channels


class TSNet(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.scatter = Scatter_layer(
            in_channels=in_channels,
            trainable_f=kwargs.get("trainable_f", False),
            max_graph_size=kwargs.get("max_graph_size", 512),
        )
        self.lin = Linear(self.scatter.out_shape(), out_channels)
        self.act = nn.LeakyReLU()

    def forward(self, data):
        x = self.scatter(data)
        x = self.act(x)
        return self.lin(x)

    def loss_function(self, predictions, targets, valid_step=False):
        _, y_true = targets
        loss = nn.MSELoss()(predictions.flatten(), y_true.flatten())
        return loss, {"loss": loss}
