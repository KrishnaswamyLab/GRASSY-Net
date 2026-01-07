import numpy as np
import torch
from torch.nn import Linear
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

device = torch.device("cuda")

## Author: Alex Tong
## Reference: Data-Driven Learning of Geometric Scattering Networks, IEEE Machine Learning for Signal Processing Workshop 2021

def scatter_moments(graph, batch_indices, moments_returned=4):
    
    """ Compute specified statistical coefficients for each feature of each graph passed. 
        The graphs expected are disjoint subgraphs within a single graph, whose feature tensor is passed as argument "graph."
        "batch_indices" connects each feature tensor to its home graph.
        "Moments_returned" specifies the number of statistical measurements to compute. 
        If 1, only the mean is returned. If 2, the mean and variance. If 3, the mean, variance, and skew. If 4, the mean, variance, skew, and kurtosis.
        The output is a dictionary. You can obtain the mean by calling output["mean"] or output["skew"], etc.
    """

    # Step 1: Aggregate the features of each mini-batch graph into its own tensor
    graph_features = [torch.zeros(0).to(graph) for i in range(torch.max(batch_indices) + 1)]

    for i, node_features in enumerate(graph):

        # Sort the graph features by graph, according to batch_indices. For each graph, create a tensor whose first row is the first element of each feature, etc.
        # print("node features are", node_features)
        
        if (len(graph_features[batch_indices[i]]) == 0):  
            # If this is the first feature added to this graph, fill it in with the features.
            graph_features[batch_indices[i]] = node_features.view(-1, 1, 1)  # .view(-1,1,1) changes [1,2,3] to [[1],[2],[3]], so that we can add each column to the respective row.
        else:
            graph_features[batch_indices[i]] = torch.cat((graph_features[batch_indices[i]], node_features.view(-1, 1, 1)), dim=1)  # concatenates along columns

    statistical_moments = {"mean": torch.zeros(0).to(graph)}

    if moments_returned >= 2:
        statistical_moments["variance"] = torch.zeros(0).to(graph)
    if moments_returned >= 3:
        statistical_moments["skew"] = torch.zeros(0).to(graph)
    if moments_returned >= 4:
        statistical_moments["kurtosis"] = torch.zeros(0).to(graph)

    for data in graph_features:

        data = data.squeeze()
        
        def m(i):  # ith moment, computed with derivation data
            return torch.mean(deviation_data ** i, axis=1)

        mean = torch.mean(data, dim=1, keepdim=True)
        
        if moments_returned >= 1:
            statistical_moments["mean"] = torch.cat(
                (statistical_moments["mean"], mean.T), dim=0
            )

        # produce matrix whose every row is data row - mean of data row

        #for a in mean:
        #    mean_row = torch.ones(data.shape[1]).to( * a
        #    tuple_collect.append(
        #        mean_row[None, ...]
        #    )  # added dimension to concatenate with differentiation of rows
        # each row contains the deviation of the elements from the mean of the row
        
        deviation_data = data - mean
        
        # variance: difference of u and u mean, squared element wise, summed and divided by n-1
        variance = m(2)
        
        if moments_returned >= 2:
            statistical_moments["variance"] = torch.cat(
                (statistical_moments["variance"], variance[None, ...]), dim=0
            )

        # skew: 3rd moment divided by cubed standard deviation (sd = sqrt variance), with correction for division by zero (inf -> 0)
        # add small eps to denominator to avoid division-by-zero and clamp extreme values
        eps = 1e-8
        skew = m(3) / ((variance + eps) ** (3 / 2))
        skew[skew > 1e6] = 0
        skew[skew != skew] = 0
        if moments_returned >= 3:
            statistical_moments["skew"] = torch.cat(
                (statistical_moments["skew"], skew[None, ...]), dim=0
            )

        # kurtosis: fourth moment, divided by variance squared. Using Fischer's definition to subtract 3 (default in scipy)
        kurtosis = m(4) / ((variance + eps) ** 2) - 3
        kurtosis[kurtosis > 1e6] = -3
        kurtosis[kurtosis != kurtosis] = -3
        if moments_returned >= 4:
            statistical_moments["kurtosis"] = torch.cat(
                (statistical_moments["kurtosis"], kurtosis[None, ...]), dim=0
            )
    
    # Concatenate into one tensor (alex)
    statistical_moments = torch.cat([v for k,v in statistical_moments.items()], axis=1)
    #statistical_moments = torch.cat([statistical_moments['mean'],statistical_moments['variance']],axis=1)
    
    return statistical_moments


class LazyLayer(torch.nn.Module):
    
    """ Currently a single elementwise multiplication with one laziness parameter per
    channel. this is run through a softmax so that this is a real laziness parameter
    """

    def __init__(self, n):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.Tensor(2, n))

    def forward(self, x, propogated):
        inp = torch.stack((x, propogated), dim=1)
        s_weights = torch.nn.functional.softmax(self.weights, dim=0)
        return torch.sum(inp * s_weights, dim=-2)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weights)
    

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, add_self_loops=False, dtype=None):

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                 device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, 1, num_nodes)
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    
    return edge_index, deg_inv_sqrt[row] * edge_weight


class Diffuse(MessagePassing):

    """ Implements low pass walk with optional weights
    """

    def __init__(self, in_channels, out_channels, trainable_laziness=False, fixed_weights=True):

        super().__init__(aggr="add", node_dim=-3)  # "Add" aggregation.
        assert in_channels == out_channels
        self.trainable_laziness = trainable_laziness
        self.fixed_weights = fixed_weights
        if trainable_laziness:
            self.lazy_layer = LazyLayer(in_channels)
        if not self.fixed_weights:
            self.lin = torch.nn.Linear(in_channels, out_channels)


    def forward(self, x, edge_index, edge_weight=None):

        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 2: Linearly transform node feature matrix.
        # turn off this step for simplicity
        if not self.fixed_weights:
            x = self.lin(x)

        # Step 3: Compute normalization
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(self.node_dim), dtype=x.dtype)

        # Step 4-6: Start propagating messages.
        propogated = self.propagate(
            edge_index, edge_weight=edge_weight, size=None, x=x,
        )
        if not self.trainable_laziness:
            return 0.5 * (x + propogated)

        return self.lazy_layer(x, propogated)


    def message(self, x_j, edge_weight):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return edge_weight.view(-1, 1, 1) * x_j


def feng_filters():
    results = [4]
    for i in range(2, 4):
        for j in range(0, i):
            results.append(4*i+j)

    return results


class Scatter(torch.nn.Module):
    def __init__(self, in_channels, max_graph_size, trainable_scattering=False):

        super().__init__()
        self.in_channels = in_channels
        self.trainable_scattering = trainable_scattering
        
        self.diffusion_layer1 = Diffuse(in_channels, in_channels, trainable_scattering)
        self.diffusion_layer2 = Diffuse(
            4 * in_channels, 4 * in_channels, trainable_scattering
        )

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
            requires_grad=trainable_scattering,
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


    def forward(self, data):

        x, edge_index = data.x, data.edge_index
        s0 = x[:,:,None]
        avgs = [s0]
        # import pdb; pdb.set_trace()
        for i in range(16):
            avgs.append(self.diffusion_layer1(avgs[-1], edge_index))
        for j in range(len(avgs)):
            avgs[j] = avgs[j][None, :, :, :]  # add an extra dimension to each tensor to avoid data loss while concatenating TODO: is there a faster way to do this?
        
        # Combine the diffusion levels into a single tensor.
        diffusion_levels = torch.cat(avgs)
        
        # Reshape the 3d tensor into a 2d tensor and multiply with the wavelet_constructor matrix
        # This simulates the below subtraction:
        # filter1 = avgs[1] - avgs[2]
        # filter2 = avgs[2] - avgs[4]
        # filter3 = avgs[4] - avgs[8]
        # filter4 = avgs[8] - avgs[16]
        subtracted = torch.matmul(self.wavelet_constructor, diffusion_levels.view(17, -1))
        subtracted = subtracted.view(4, x.shape[0], x.shape[1]) # reshape into given input shape
        
        # Replace ABSOLUTE with per-wavelet MLPs for s1
        # subtracted shape: (4, N, C) -> need to apply MLP per wavelet scale
        # REPLACED: s1 = torch.abs(
        # torch.transpose(torch.transpose(subtracted, 0, 1), 1, 2))  # transpose the dimensions to match previous
        s1_list = []
        for l in range(self.L):
            # subtracted[l] has shape (N, C)
            s1_list.append(self.mlps[l](subtracted[l]))  # (N, C)
        s1 = torch.stack(s1_list, dim=2)  # (N, C, L) to match original s1 shape

        # perform a second wave of diffusing, on the recently diffused.
        avgs = [s1]
        for i in range(16): # diffuse over diffusions
            avgs.append(self.diffusion_layer2(avgs[-1], edge_index))
        for i in range(len(avgs)): # add an extra dimension to each diffusion level for concatenation
            avgs[i] = avgs[i][None, :, :, :]
        diffusion_levels2 = torch.cat(avgs)
        # import pdb; pdb.set_trace()
        # Having now generated the diffusion levels, we can cmobine them as before
        subtracted2 = torch.matmul(self.wavelet_constructor, diffusion_levels2.view(17, -1))
        subtracted2 = subtracted2.view(4, s1.shape[0], s1.shape[1], s1.shape[2])  # reshape into given input shape
        subtracted2 = torch.transpose(subtracted2, 0, 1)

        # SUBSTITUTED subtracted2 = torch.abs(subtracted2.reshape(-1, self.in_channels, 4))
        subtracted2 = subtracted2.reshape(-1, self.in_channels, 4)  # (N*L, C, 4)
            # Apply MLPs across the last dimension (the 4 wavelet scales)
        s2_list = []
        for l in range(self.L):
            # subtracted2[:, :, l] has shape (N*L, C)
            s2_list.append(self.mlps[l](subtracted2[:, :, l]))
        subtracted2 = torch.stack(s2_list, dim=2)  # (N*L, C, 4)

        s2_swapped = torch.reshape(torch.transpose(subtracted2, 1, 2), (-1, 16, self.in_channels))
        s2 = s2_swapped[:, feng_filters()]
        # import pdb; pdb.set_trace()
        x = torch.cat([s0, s1], dim=2)
        x = torch.transpose(x, 1, 2)
        x = torch.cat([x, s2], dim=1)
        # import pdb; pdb.set_trace()
        #x = scatter_mean(x, batch, dim=0)
        if getattr(data, 'batch', None) is not None:
            x = scatter_moments(x, data.batch, 4)
        else:
            # handle single-graph case: assign all nodes to batch 0
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=x.device)
            x = scatter_moments(x, batch, 4)
        # if hasattr(data, 'batch'):
        #     x = scatter_moments(x, data.batch, 4)
        # else:
        #     # import pdb; pdb.set_trace()
        #     x = scatter_moments(x, torch.zeros(data.x.shape[0], dtype=torch.int32), 4)
            # print('x returned shape', x.shape)
        return x, self.wavelet_constructor

    def out_shape(self):

        # x * 4 moments * in
        return 11 * 4 * self.in_channels


class TSNet(torch.nn.Module):

    def __init__(self, in_channels, out_channels, edge_in_channels = None, trainable_scattering=False, **kwargs):

        max_graph_size = kwargs.get("max_graph_size", 128)
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_in_channels = edge_in_channels
        self.trainable_scattering = trainable_scattering
        self.scatter = Scatter(in_channels, max_graph_size=max_graph_size, trainable_scattering=trainable_scattering)
        self.lin1 = Linear(self.scatter.out_shape(), out_channels)
        self.act = torch.nn.LeakyReLU()


    def forward(self, data):

        x, sc = self.scatter(data)
        x = self.act(x)
        x = self.lin1(x)
        return x, sc
