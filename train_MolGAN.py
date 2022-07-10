import os
from collections import OrderedDict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import deepchem as dc
import deepchem.models
from deepchem.models import BasicMolGANModel as MolGAN
from deepchem.models.optimizers import ExponentialDecay
from deepchem.feat.molecule_featurizers.molgan_featurizer import GraphMatrix

import tensorflow as tf
from tensorflow import one_hot

from rdkit import Chem

physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

# process smile strings
num_atoms = 100
tranche_dat = np.load(os.path.join("datasets", "BBAB_subset.npy"), allow_pickle=True).item()
smiles = tranche_dat.keys()
filtered_smiles = [x for x in smiles if Chem.MolFromSmiles(x).GetNumAtoms() < num_atoms]

# create featurizer
feat = dc.feat.MolGanFeaturizer(max_atom_count=num_atoms, 
                            atom_labels=[0, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 19, 20, 24, 29, 35])
features = feat.featurize(filtered_smiles)

# remove invalid molecules
indices = [i for i, data in enumerate(features) if type(data) is GraphMatrix]
features = [features[i] for i in indices]

# initialize model
gan = MolGAN(learning_rate=ExponentialDecay(0.001, 0.9, 5000), vertices=num_atoms)
dataset = dc.data.NumpyDataset([x.adjacency_matrix for x in features],[x.node_features for x in features])

def iterbatches(epochs):
    for i in range(epochs):
        for batch in dataset.iterbatches(batch_size=gan.batch_size, pad_batches=True):
            adjacency_tensor = one_hot(batch[0], gan.edges)
            node_tensor = one_hot(batch[1], gan.nodes)
            yield {gan.data_inputs[0]: adjacency_tensor, gan.data_inputs[1]:node_tensor}

gan.fit_gan(iterbatches(25), generator_steps=0.2, checkpoint_interval=5000)

# generate 1000 molecules
generated_data = gan.predict_gan_generator(1000)
nmols = feat.defeaturize(generated_data)
print("{} molecules generated".format(len(nmols)))

# filter invalid molecules
nmols = list(filter(lambda x: x is not None, nmols))
print ("{} valid molecules".format(len(nmols)))

# remove duplicate molecules
nmols_smiles = [Chem.MolToSmiles(m) for m in nmols]
nmols_smiles_unique = list(OrderedDict.fromkeys(nmols_smiles))
nmols_uniq = [Chem.MolFromSmiles(x) for x in nmols_smiles_unique]
print ("{} unique valid molecules".format(len(nmols_uniq)))