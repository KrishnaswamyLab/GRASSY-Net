from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch_geometric.data
from torch.nn import Linear
from torch_scatter import scatter_mean
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
import networkx as nx
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from molecule import Molecule

#from LEGS_module import Scatter

#from torch_geometric.utils.convert import from_networkx

import networkx as nx
from pysmiles import read_smiles


class BindingDB(Dataset):
    """BindingDB dataset."""

    def __init__(self, file_name, transform=None):
        tranch = np.load(file_name, allow_pickle=True)
        
        self.tranch = tranch.item()
        self.smi = list(tranch.item().keys())
        self.props = list(tranch.item().values())
        self.transform = transform
        self.num_node_features = 1
        self.num_classes = 5
        #self.atom_types = ['C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I']
        #self.atom_types = [0, 1, 2, 3, 4, 5, 6, 7]
        #self.id2atom = [6, 7, 8, 9, 16, 17, 35, 53]
        self.atom_types = [6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
        self.num_atom_type = 10
        self.kekulize=True
        self.num_bond_type = 3

    def __len__(self):
        
        return len(self.tranch)

    def _standarize_index(self, index, count):
        if isinstance(index, slice):
            start = index.start or 0
            if start < 0:
                start += count
            stop = index.stop or count
            if stop < 0:
                stop += count
            step = index.step or 1
            index = range(start, stop, step)
        elif not isinstance(index, list):
            raise ValueError("Unknown index `%s`" % index)
        return index

    def get_item(self, index):
        
        smi = self.smi[index]
        item = {"graph": Molecule.from_smiles(smi)}
    
        item.update({'qed': self.tranch[smi]['qed']})

        if self.transform:
            item = self.transform(item)
        return item

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.get_item(index)

        index = self._standarize_index(index, len(self))
        return [self.get_item(i) for i in index]

#     def __getitem__(self, idx):
             
        
#         smi = self.smi[idx]
            
#         prop_dict = self.tranch[smi]
        
#         props = np.zeros(5)

#         props[0] = prop_dict['MinEStateIndex']
#         props[1] = prop_dict['MolWt']
#         #props[2] = prop_dict['MinPartialCharge']
#         props[2] = prop_dict['qed']
#         props[3] = prop_dict['TPSA']
# #         props[4] = prop_dict['MolMR']
# #         props[5] = prop_dict['MolLogP']
# #         props[6] = prop_dict['FpDensityMorgan1']
# #         props[7] = prop_dict['Chi3v']
# #         props[8] = prop_dict['Chi0']
#         props[4] = prop_dict['k_i']
# #         props[9] = prop_dict['EState_VSA1']
# #         props[10] = prop_dict['qed']
# #         props[11] = prop_dict['mwt']
# #         props[12] = prop_dict['PEOE_VSA8']
# #         props[13] = prop_dict['FractionCSP3']
# #         props[14] = prop_dict['MaxEStateIndex']
# #         props[15] = prop_dict['Kappa3']
# #         props[16] = prop_dict['Kappa1']
# #         props[17] = prop_dict['Chi0n']
# #         props[18] = prop_dict['MaxPartialCharge']
# #         props[19] = prop_dict['BertzCT']
        
#         mol = read_smiles(smi)

#         data = from_networkx_custom(mol)
#         adj = nx.adjacency_matrix(mol).todense()
#         if data.num_nodes >= 32:
#             data.adj = torch.tensor(adj[:32, :32]).float()
#         else:
#             data.adj = torch.nn.ConstantPad2d((0, 32-data.num_nodes, 0, 32-data.num_nodes), 0)(torch.tensor(adj)).float()

        
#         data.y = torch.tensor([props]).float()
        
#         feats = []
#         for i in range(data.num_nodes):
# #             arr = np.zeros(4)
# #             arr[0] = 1.
# #             arr[1] = 1. if data.element[i] == 'C' else 0.
# #             arr[2] = 1. if data.element[i] == 'O' else 0.
# #             arr[3] = 1. if data.element[i] == 'N' else 0.
# #             feats.append(arr)
#             feats.append([1.])
#         labels = torch.zeros((32, 6)).float()
#         max = 32 if data.num_nodes > 32 else data.num_nodes
#         for i in range(max):
#             labels[i][0] = 1. if data.element[i] == 'C' else 0.
#             labels[i][1] = 1. if data.element[i] == 'N' else 0.
#             labels[i][2] = 1. if data.element[i] == 'O' else 0.
#             labels[i][3] = 1. if data.element[i] == 'F' else 0.
#             labels[i][4] = 1. if data.element[i] == 'Cl' else 0.
#             labels[i][5] = 1. if data.element[i] == 'S' else 0.

#         data.x = torch.tensor(feats)
#         data.L = labels

#         data.edge_attr = None

#         if self.transform: 
#             return self.transform(data)
#         else:
#             #return data
#             return data.adj, data.L


class Scattering(object):

    def __init__(self):
        model = Scatter(1, trainable_laziness=None)
        model.load_state_dict(torch.load("/home/jacksongrady/graphGeneration/gsae/gsae/LEGS/results/BindingBD_LEGS_dyadic_model.npy"))
        model.eval()
        self.model = model
    
    def __call__(self, sample):
        props = sample.y

        elements = sample.element
        carbon_arr =[]
        oxy_arr = []
        nitro_arr = []
        for entry in elements:
            if entry == 'C':
                carbon_arr.append([1.])
            else:
                carbon_arr.append([0.]) 
            if entry == 'O':
                oxy_arr.append([1.])
            else:
                oxy_arr.append([0.])
            if entry == 'N':
                nitro_arr.append([1.])
            else:
                nitro_arr.append([0.])

        carbon_arr = torch.Tensor(carbon_arr)
        oxy_arr = torch.Tensor(oxy_arr)
        nitro_arr = torch.Tensor(nitro_arr)

        scat = self.model(sample)
        sample.x = carbon_arr
        carb = self.model(sample)
        sample.x = oxy_arr
        oxy = self.model(sample)
        sample.x = nitro_arr
        nitro = self.model(sample)

        appended = torch.cat((scat[0][0].detach(), carb[0][0].detach(), oxy[0][0].detach(), nitro[0][0].detach()), 0)
        
        return appended, props[0]
        
        
def from_networkx_custom(G):
    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """
    import networkx as nx

    G = nx.convert_node_labels_to_integers(G)
    G = G.to_directed() if not nx.is_directed(G) else G
    edge_index = torch.LongTensor(list(G.edges)).t().contiguous()

    data = {}

    for i, (_, feat_dict) in enumerate(G.nodes(data=True)):
        for key, value in feat_dict.items():
            if(str(key) != "stereo"):
                data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for i, (_, _, feat_dict) in enumerate(G.edges(data=True)):
        for key, value in feat_dict.items():
            data[str(key)] = [value] if i == 0 else data[str(key)] + [value]

    for key, item in data.items():
        try:
            data[key] = torch.tensor(item)
        except ValueError:
            pass

    data['edge_index'] = edge_index.view(2, -1)
    data = torch_geometric.data.Data.from_dict(data)
    data.num_nodes = G.number_of_nodes()

    return data


def scatter_moments(graph, batch_indices, moments_returned=4):
    """ Compute specified statistical coefficients for each feature of each graph passed. The graphs expected are disjoint subgraphs within a single graph, whose feature tensor is passed as argument "graph."
        "batch_indices" connects each feature tensor to its home graph.
        "Moments_returned" specifies the number of statistical measurements to compute. If 1, only the mean is returned. If 2, the mean and variance. If 3, the mean, variance, and skew. If 4, the mean, variance, skew, and kurtosis.
        The output is a dictionary. You can obtain the mean by calling output["mean"] or output["skew"], etc."""
    # Step 1: Aggregate the features of each mini-batch graph into its own tensor
    graph_features = [torch.zeros(0).to(graph) for i in range(torch.max(batch_indices) + 1)]
    for i, node_features in enumerate(
        graph
    ):  # Sort the graph features by graph, according to batch_indices. For each graph, create a tensor whose first row is the first element of each feature, etc.
        #        print("node features are",node_features)
        if (
            len(graph_features[batch_indices[i]]) == 0
        ):  # If this is the first feature added to this graph, fill it in with the features.
            graph_features[batch_indices[i]] = node_features.view(
                -1, 1, 1
            )  # .view(-1,1,1) changes [1,2,3] to [[1],[2],[3]],so that we can add each column to the respective row.
        else:
            graph_features[batch_indices[i]] = torch.cat(
                (graph_features[batch_indices[i]], node_features.view(-1, 1, 1)), dim=1
            )  # concatenates along columns

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
        skew = m(3) / (variance ** (3 / 2))
        skew[
            skew > 1000000000000000
        ] = 0  # multivalued tensor division by zero produces inf
        skew[
            skew != skew
        ] = 0  # single valued division by 0 produces nan. In both cases we replace with 0.
        if moments_returned >= 3:
            statistical_moments["skew"] = torch.cat(
                (statistical_moments["skew"], skew[None, ...]), dim=0
            )

        # kurtosis: fourth moment, divided by variance squared. Using Fischer's definition to subtract 3 (default in scipy)
        kurtosis = m(4) / (variance ** 2) - 3
        kurtosis[kurtosis > 1000000000000000] = -3
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
        mult = inp * s_weights
        return torch.sum(mult, dim=-2)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weights)
        
def gcn_norm(edge_index, edge_weight=None, num_nodes=None,
             add_self_loops=False, dtype=None):
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

    def __init__(
        self, in_channels, out_channels, trainable_laziness=False, fixed_weights=True
    ):
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
    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)


    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        # Step 6: Return new node embeddings.
        return aggr_out


def feng_filters():
    tmp = np.arange(16).reshape(4,4) #tmp doesn't seem to be used!
    results = [4]
    for i in range(2, 4):
        for j in range(0, i):
            results.append(4*i+j)
    return results


class Scatter(torch.nn.Module):
    def __init__(self, in_channels, trainable_laziness=False):
        super().__init__()
        self.in_channels = in_channels
        self.trainable_laziness = trainable_laziness
        self.diffusion_layer1 = Diffuse(in_channels, in_channels, trainable_laziness)
        self.diffusion_layer2 = Diffuse(
            4 * in_channels, 4 * in_channels, trainable_laziness
        )
        
        #self.wavelet_constructor = torch.nn.Parameter(torch.rand(4, 17))
        self.wavelet_constructor = torch.nn.Parameter(torch.tensor([
            [0, -1.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1]
        ], requires_grad=True))
        
#         self.wavelet_constructor = torch.nn.Parameter(torch.tensor([
#             w0,
#             w1,
#             w2,
#             w3
# #             [0, 0, -1.0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# #             [0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
# #             [0, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
# #             [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 1]
#          ], requires_grad=True)).float()
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        s0 = x[:,:,None]
        avgs = [s0]
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
        s1 = torch.abs(
            torch.transpose(torch.transpose(subtracted, 0, 1), 1, 2))  # transpose the dimensions to match previous

        # perform a second wave of diffusing, on the recently diffused.
        avgs = [s1]
        for i in range(16): # diffuse over diffusions
            avgs.append(self.diffusion_layer2(avgs[-1], edge_index))
        for i in range(len(avgs)): # add an extra dimension to each diffusion level for concatenation
            avgs[i] = avgs[i][None, :, :, :]
        diffusion_levels2 = torch.cat(avgs)
        # Having now generated the diffusion levels, we can cmobine them as before
        subtracted2 = torch.matmul(self.wavelet_constructor, diffusion_levels2.view(17, -1))
        subtracted2 = subtracted2.view(4, s1.shape[0], s1.shape[1], s1.shape[2])  # reshape into given input shape
        subtracted2 = torch.transpose(subtracted2, 0, 1)
        subtracted2 = torch.abs(subtracted2.reshape(-1, self.in_channels, 4))
        s2_swapped = torch.reshape(torch.transpose(subtracted2, 1, 2), (-1, 16, self.in_channels))
        s2 = s2_swapped[:, feng_filters()]

        x = torch.cat([s0, s1], dim=2)
        x = torch.transpose(x, 1, 2)
        x = torch.cat([x, s2], dim=1)
        # x = scatter_mean(x, batch, dim=0)
        if hasattr(data, 'batch'):
            x = scatter_moments(x, data.batch, 4)
        else:
            x = scatter_moments(x, torch.zeros(data.x.shape[0], dtype=torch.int32), 4)
        # print('x returned shape', x.shape)
        return x, self.wavelet_constructor

    def out_shape(self):
        # x * 4 moments * in
        return 11 * 4 * self.in_channels
        
    