import numpy as np
import pandas as pd
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd()))+'/mol2vec')

from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole

from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec

data = np.load("datasets/BBAB_subset.npy", allow_pickle=True).item()

model = word2vec.Word2Vec.load('mol2vec/examples/models/model_300dim.pkl')

smis = data.keys()
aas = [Chem.MolFromSmiles(x) for x in smis]

aa_sentences = [mol2alt_sentence(x, 1) for x in aas]

df_vec = pd.DataFrame()
df_vec['sentence'] = aa_sentences
sentences2vec(aa_sentences, model, unseen='UNK')

print(df_vec.shape)
