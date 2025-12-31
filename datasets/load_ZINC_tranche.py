from __future__ import print_function, division

import os, math, torch, pathlib

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data

import torch_geometric.data
from torch_geometric import utils
from torch_geometric import data

from pysmiles import read_smiles

from models.LEGS_module import Scatter
from rdkit import Chem
from torch_geometric.utils import from_networkx

def read_smiles_rdkit(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    return nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol))


class ZINCDataset(Dataset):

    """ZINC Tranch data"""

    def __init__(self, file_name, transform=None, prop_stat_dict=None, include_ki=False):
        

        self.prop_list = ['qed', 'HeavyAtomMolWt', 'MolWt', 'BalabanJ', 'BertzCT', 'Ipc', 'TPSA', 'NumHAcceptors', 'NumHDonors', 'RingCount', 'MolLogP', 'SAscore', 'FSP3'] # new properites

        if include_ki:
            self.prop_list.append('Ki')
        
        self.tranch = np.load(file_name, allow_pickle=True).item()
        
        if prop_stat_dict != None:
            self.stats = np.load(prop_stat_dict, allow_pickle=True).item()
        else:
            self.stats = None

        self.transform = transform
        self.num_node_features = 14 # changed for 8 atoms but only the previous pairs. not sure if we need to add the new pairs. 
        self.num_classes = len(self.prop_list)
        self.smi = list(self.tranch.keys())


    def __len__(self):
        
        return len(self.smi)

    
    def __getitem__(self, idx):         
        
        smi = self.smi[idx]

        props = np.zeros(self.num_classes)
        no_zscore = np.zeros(self.num_classes)

        if self.stats != None:
            #we want to zscore
            for i, entry in enumerate(self.prop_list):
                prop_value = self.tranch[smi][entry]
                z_scored = (prop_value - self.stats[entry]['mean']) / self.stats[entry]['std']
                props[i] = z_scored
        else:
            for i, entry in enumerate(self.prop_list):
                prop_value = self.tranch[smi][entry]
                props[i] = prop_value
                no_zscore[i] = prop_value

        # mol = read_smiles(smi)
        # mol = read_smiles_rdkit(smi)
        # data = from_networkx_custom(mol)
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")

        data = mol_to_pyg(mol)

        # store properties as proper float tensors (shape [1, num_classes]) to avoid slow list->tensor conversions
        data.no_zscore_props = torch.tensor(no_zscore, dtype=torch.float32)
        data.y = torch.tensor(props, dtype=torch.float32).unsqueeze(0) # <- changed for efficiency
        # import pdb; pdb.set_trace()
        #place node features
        node_feats = []
    
        for i, entry in enumerate(data.element): 

            node_feat = np.zeros(self.num_node_features)
            
            #one hot encoding of atoms (8 types: C, O, N, S, F, Cl, Br, I)
            atom_type_map = {'C': 0, 'O': 1, 'N': 2, 'S': 3, 'F': 4, 'Cl': 5, 'Br': 6, 'I': 7}
            if entry in atom_type_map:
                node_feat[atom_type_map[entry]] = 1.

            #pair encoding of atoms (keep existing 6 pairs)
            if entry == 'C' or entry == 'O':
                node_feat[8] = 1.  # C-O pair
            if entry == 'C' or entry == 'N':
                node_feat[9] = 1.  # C-N pair
            if entry == 'C' or entry == 'S':
                node_feat[10] = 1.  # C-S pair
            if entry == 'O' or entry == 'N':
                node_feat[11] = 1.  # O-N pair
            if entry == 'O' or entry == 'S':
                node_feat[12] = 1.  # O-S pair
            if entry == 'N' or entry == 'S':
                node_feat[13] = 1.  # N-S pair
            # didnt add pairs for the new atoms for now. 

            node_feats.append(node_feat)

        data.x = torch.tensor(np.array(node_feats), dtype=torch.float32) # <- same for efficiency
        # import pdb; pdb.set_trace()
        if self.transform: 
            return self.transform(data)
        else:
            return data


class Scattering(object):

    def __init__(self, scatter_model_name=None):

        model = Scatter(14, trainable_laziness=None) # 14 rather than 10,new atoms but not new pairs 
        if scatter_model_name == None:
            raise ValueError("Please specify a pretrained scatter module. If you'd like to use an untrained model, specify\
            scatter_model_name='untrained'. Otherwise, use the .npy file of the model")
        elif scatter_model_name != 'untrained':
            # Load to CPU first (works regardless of where model was saved)
            state_dict = torch.load(scatter_model_name, map_location='cpu')
            model.load_state_dict(state_dict)
            model = model.cpu()
        model.eval()
        self.model = model
            
    def __call__(self, sample):

        props = sample.y
        to_return = self.model(sample)
        
        return to_return[0][0].detach(), sample.y[0]



def mol_to_pyg(mol):
    """
    Converts an RDKit molecule to a PyTorch Geometric Data object.
    """

    # --- 1️⃣ Extract element symbols ---
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # --- 2️⃣ Create edge index (both directions for undirected graphs) ---
    edge_index = []
    edge_weight = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append((i, j))
        edge_index.append((j, i))
        edge_weight.append(bond.GetBondTypeAsDouble())
        edge_weight.append(bond.GetBondTypeAsDouble())

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)

    # --- 3️⃣ One-hot encode atom types ---
    # unique_atoms = ['C', 'O', 'N', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'H']
    # node_feats = []
    # for e in elements:
    #     feat = [1.0 if e == u else 0.0 for u in unique_atoms]
    #     node_feats.append(feat)
    # x = torch.tensor(node_feats, dtype=torch.float32)

    # --- 4️⃣ Assemble Data object ---
    data = Data(
        # x=x,
        edge_index=edge_index,
        weight=edge_weight,
        num_nodes=mol.GetNumAtoms(),
    )

    # --- 5️⃣ Store element labels for convenience ---
    data.element = elements

    return data

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
