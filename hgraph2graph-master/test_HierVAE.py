import os, sys, math, random
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import rdkit
from rdkit import Chem

import hgraph
from hgraph import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

class Args():
    def __init__(self):
        self.vocab = 'ZINC_FBAB_vocab.txt'
        self.atom_vocab = common_atom_vocab
        self.model = 'ckpt/ZINC-FBAB-pretrained/model.ckpt.1500'
        self.seed = 13
        self.nsample = 100
        self.rnn_type = 'LSTM'
        self.hidden_size = 250
        self.embed_size = 250
        self.batch_size = 50
        self.latent_size = 32
        self.depthT = 15
        self.depthG = 15
        self.diterT = 1
        self.diterG = 3
        self.dropout = 0.0

args = Args()

vocab = [x.strip("\r\n ").split() for x in open("ZINC_FBAB_vocab.txt")]
args.vocab = PairVocab(vocab, cuda=True)

model = HierVAE(args).to(device)
if not torch.cuda.is_available() == 'cpu':
    model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu'))[0])
else:
    model.load_state_dict(torch.load(args.model)[0])
model.eval()

torch.manual_seed(args.seed)
random.seed(args.seed)

nmols_smiles = []

with torch.no_grad():
    for _ in range(args.nsample // args.batch_size):
        smiles_list = model.sample(args.batch_size, greedy=True)
        for _,smiles in enumerate(smiles_list):
            nmols_smiles.append(smiles)

print ("{} valid molecules".format(len(nmols_smiles)))

nmols_smiles_unique = list(OrderedDict.fromkeys(nmols_smiles))
print ("{} unique valid molecules".format(len(nmols_smiles_unique)))

train_smi = []
with open("../datasets/FBAB.txt",'r') as f:
    for line in f:
	    train_smi.append(line.strip())

nmols_smiles_novel = list()
for smi in nmols_smiles_unique:
    if smi not in train_smi:
        nmols_smiles_novel.append(smi)
print ("{} novel valid molecules".format(len(nmols_smiles_novel)))

with open('ZINC_FBAB_gen.txt', 'w') as f:
    for smi in list(nmols_smiles_novel):
        f.write(f"{smi}\n")