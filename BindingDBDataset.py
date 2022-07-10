from __future__ import print_function, division

import os

import numpy as np
import networkx as nx

from torch.utils.data import Dataset, DataLoader

from rdkit import Chem, RDLogger
from molecule import Molecule

class BindingDB(Dataset):

    """BindingDB dataset"""

    def __init__(self, file_name, transform=None):

        tranch = np.load(file_name, allow_pickle=True)
        
        self.tranch = tranch.item()
        self.smi = list(tranch.item().keys())
        self.props = list(tranch.item().values())
        self.transform = transform
        self.num_node_features = 1
        self.num_classes = 5
        self.atom_types = ['C', 'N', 'O', 'F', 'Cl', 'S', 'Br', 'I']
        self.atom_types = [ 6,   7,   8,   9,  17,   16,   35,   53]
        self.num_atom_type = 8
        self.kekulize = True
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
        mol  = Molecule.from_smiles(smi)
        mol.bond_stereo[:] = 0
        item = {"graph": mol}
        item.update({'qed': self.tranch[smi]['qed']})

        if self.transform:
            item = self.transform(item)
        return item

    def __getitem__(self, index):

        if isinstance(index, int):
            return self.get_item(index)

        index = self._standarize_index(index, len(self))
        return [self.get_item(i) for i in index]