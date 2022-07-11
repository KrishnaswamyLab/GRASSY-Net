import os, sys, time, random, pickle

import numpy as np
import pandas as pd
import networkx as nx

import selfies as sf
import deepchem as dc

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import DataStructs

from pysmiles import read_smiles
from fcd import get_fcd, load_ref_model,canonical_smiles, get_predictions, calculate_frechet_distance

import logging
logging.getLogger('pysmiles').setLevel(logging.CRITICAL)
logging.getLogger('rdkit').setLevel(logging.CRITICAL)

all_RDKit_Descriptors = [desc[0] for desc in Descriptors.descList]
selected_RDKit_Descriptors = ['qed', 'ExactMolWt', 'MaxPartialCharge', 'MinPartialCharge', 'MolLogP', 'TPSA', ]

def tanimoto_similarity(smi1, smi2):

    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=2048)
    s = round(DataStructs.TanimotoSimilarity(fp1,fp2),3)
    
    return s

def fcd_score(smi_list_1, smi_list_2):

    model = load_ref_model()

    can_sample1 = [w for w in canonical_smiles(smi_list_1) if w is not None]
    can_sample2 = [w for w in canonical_smiles(smi_list_2) if w is not None]

    act1 = get_predictions(model, can_sample1)
    act2 = get_predictions(model, can_sample2)

    mu1 = np.mean(act1, axis=0)
    sigma1 = np.cov(act1.T)

    mu2 = np.mean(act2, axis=0)
    sigma2 = np.cov(act2.T)

    fcd_score = calculate_frechet_distance(
        mu1=mu1,
        mu2=mu2, 
        sigma1=sigma1,
        sigma2=sigma2)

    return fcd_score

def smi2selfie(smi):

    try:
        return sf.encoder(smi)
    except:
        return None

def selfie2smi(selfie):

    try:
        return sf.decoder(selfie)
    except:
        return None
        
def smi2img(smi_list):

    smiimgfeaturizer = dc.feat.SmilesToImage(img_size=256, img_spec='std')
    return smiimgfeaturizer.featurize(smi_list)

def smi2props(smi):

    mol = Chem.MolFromSmiles(smi)
    if mol:
        fns = [(x,y) for x,y in Descriptors.descList if x in selected_RDKit_Descriptors]
        res = []
        for x,y in fns:
            res.append(y(mol))
        return res
    else:
        return [None]*len(selected_RDKit_Descriptors)