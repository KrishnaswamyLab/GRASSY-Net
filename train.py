import torch
import torchdrug as td
from torchdrug import data, datasets, models, tasks, core
from BindingDBDataset import BindingDB
from torchdrug.layers import distribution
from torch import nn, optim
import pickle

#train_set = BindingDB("../../data/binding_DB/binding_DB_props/DTI_BindingDB_train_complete_re_filtered_small.npy")
# train_set = datasets.QM9("~/molecule-datasets/", kekulize=True,
#                            node_feature="symbol")

# with open("QM9.pkl", "wb") as fout:
#     pickle.dump(train_set, fout)
with open("QM9.pkl", "rb") as fin:
    train_set = pickle.load(fin)


model = models.RGCN(input_dim=train_set.num_atom_type,
                    num_relation=train_set.num_bond_type,
                    hidden_dims=[256, 256, 256], batch_norm=True)

num_atom_type = train_set.num_atom_type
# add one class for non-edge
num_bond_type = train_set.num_bond_type + 1


node_prior = distribution.IndependentGaussian(torch.zeros(num_atom_type),
                                              torch.ones(num_atom_type))

edge_prior = distribution.IndependentGaussian(torch.zeros(num_bond_type),
                                              torch.ones(num_bond_type))

node_flow = models.GraphAF(model, node_prior, num_layer=12)
edge_flow = models.GraphAF(model, edge_prior, use_edge=True, num_layer=12)    



task = tasks.AutoregressiveGeneration(node_flow, edge_flow,
                                      max_node=38, max_edge_unroll=12,
                                      criterion="nll")

optimizer = optim.Adam(task.parameters(), lr = 1e-3)
solver = core.Engine(task, train_set, None, None, optimizer, gpus=(0,), batch_size=128, log_interval=10)

solver.train(num_epoch=10)
solver.save("graphgeneration/graphaf_QM9_10epoch.pkl")

from collections import defaultdict

solver.load("./graphgeneration/graphaf_QM9_10epoch.pkl")
results = task.generate(num_sample=32)
print(results.to_smiles())

