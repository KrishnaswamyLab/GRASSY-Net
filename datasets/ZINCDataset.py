from __future__ import print_function, division

import torch

import numpy as np
import networkx as nx

from torch.utils.data import Dataset
import torch_geometric.data


from rdkit import Chem

from datasets.property_utils import PROPERTIES_TO_COMPUTE
def read_smiles_rdkit(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smi}")
    return nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol))


class ZINCDataset(Dataset):
    """ZINC Tranch data"""

    def __init__(self, file_name, transform=None, prop_stat_dict=None):
        

        self.prop_list = PROPERTIES_TO_COMPUTE.copy()
        
        self.tranch = np.load(file_name, allow_pickle=True).item()
        
        if prop_stat_dict != None:
            self.stats = np.load(prop_stat_dict, allow_pickle=True).item()
        else:
            self.stats = None
        
        self.transform = transform
        self.smi = list(self.tranch.keys())
        self.atom_types = self._scan_atom_types()
        self.atom_type_map = {atom: i for i, atom in enumerate(self.atom_types)}
        self.num_node_features = len(self.atom_types)
        self.num_classes = len(self.prop_list)
        print(f"Detected {self.num_node_features} atom types: {self.atom_types}")
    
    def _scan_atom_types(self):
        """Scan all molecules to find unique atom types."""
        atom_set = set()
        for smi in self.smi:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                for atom in mol.GetAtoms():
                    atom_set.add(atom.GetSymbol())
        return sorted(list(atom_set))


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
        # Simple one-hot encoding using dynamic atom type map
        node_feats = []
        for entry in data.element:
            node_feat = np.zeros(self.num_node_features)
            if entry in self.atom_type_map:
                node_feat[self.atom_type_map[entry]] = 1.0
            node_feats.append(node_feat)
        data.x = torch.tensor(np.array(node_feats), dtype=torch.float32) # <- same for efficiency
        # import pdb; pdb.set_trace()
        if self.transform: 
            return self.transform(data)
        else:
            return data

def mol_to_pyg(mol):
    """
    Converts an RDKit molecule to a PyTorch Geometric Data object.
    """

    # ---  Extract element symbols ---
    elements = [atom.GetSymbol() for atom in mol.GetAtoms()]

    # --- Create edge index (both directions for undirected graphs) ---
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

    # --- One-hot encode atom types ---
    # unique_atoms = ['C', 'O', 'N', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'H']
    # node_feats = []
    # for e in elements:
    #     feat = [1.0 if e == u else 0.0 for u in unique_atoms]
    #     node_feats.append(feat)
    # x = torch.tensor(node_feats, dtype=torch.float32)

    # --- Assemble Data object ---
    data = torch_geometric.data.Data(
        # x=x,
        edge_index=edge_index,
        weight=edge_weight,
        num_nodes=mol.GetNumAtoms(),
    )

    # --- Store element labels for convenience ---
    data.element = elements

    return data

def from_networkx_custom(G):

    r"""Converts a :obj:`networkx.Graph` or :obj:`networkx.DiGraph` to a
    :class:`torch_geometric.data.Data` instance.

    Args:
        G (networkx.Graph or networkx.DiGraph): A networkx graph.
    """

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