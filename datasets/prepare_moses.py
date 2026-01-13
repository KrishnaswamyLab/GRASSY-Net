"""Prepare MOSES dataset for GRASSY training."""
import numpy as np
import moses
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from tqdm import tqdm

def compute_props(mol):
    """Compute properties for a molecule (same as prepare_zinc12k.py)."""
    out = {}
    try:
        out['qed'] = float(QED.qed(mol))
    except Exception:
        out['qed'] = float('nan')
    
    try:
        out['MolWt'] = float(Descriptors.MolWt(mol))
    except Exception:
        out['MolWt'] = float('nan')
    
    try:
        mw = out['MolWt']
        num_H = sum([atom.GetTotalNumHs() for atom in mol.GetAtoms()])
        out['HeavyAtomMolWt'] = mw - num_H * 1.00794 if not np.isnan(mw) else float('nan')
    except Exception:
        out['HeavyAtomMolWt'] = float('nan')
    
    try:
        out['BalabanJ'] = float(rdMolDescriptors.CalcBalabanJ(mol))
    except Exception:
        out['BalabanJ'] = float('nan')
    
    try:
        out['BertzCT'] = float(rdMolDescriptors.CalcBertzCT(mol))
    except Exception:
        out['BertzCT'] = float('nan')
    
    try:
        out['Ipc'] = float(rdMolDescriptors.CalcIpc(mol))
    except Exception:
        out['Ipc'] = float('nan')
    
    try:
        out['TPSA'] = float(rdMolDescriptors.CalcTPSA(mol))
    except Exception:
        out['TPSA'] = float('nan')
    
    try:
        out['NumHAcceptors'] = float(rdMolDescriptors.CalcNumHBA(mol))
    except Exception:
        out['NumHAcceptors'] = float('nan')
    
    try:
        out['NumHDonors'] = float(rdMolDescriptors.CalcNumHBD(mol))
    except Exception:
        out['NumHDonors'] = float('nan')
    
    try:
        out['RingCount'] = float(mol.GetRingInfo().NumRings())
    except Exception:
        out['RingCount'] = float('nan')
    
    try:
        out['MolLogP'] = float(Descriptors.MolLogP(mol))
    except Exception:
        out['MolLogP'] = float('nan')
    
    try:
        out['SAscore'] = float(rdMolDescriptors.CalcSAScore(mol))
    except Exception:
        out['SAscore'] = float('nan')
    
    try:
        out['FSP3'] = float(rdMolDescriptors.CalcFractionCsp3(mol))
    except Exception:
        out['FSP3'] = float('nan')
    
    return out

if __name__ == '__main__':
    print("Loading MOSES training set...")
    train_smiles = moses.get_dataset('train')
    print(f"Loaded {len(train_smiles)} SMILES")
    
    print("Computing properties...")
    out_dict = {}
    skipped = 0
    
    for smi in tqdm(train_smiles):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                skipped += 1
                continue
            out_dict[smi] = compute_props(mol)
        except Exception as e:
            skipped += 1
            continue
    
    # Save
    output_path = 'datasets/MOSES.npy'
    np.save(output_path, out_dict)
    print(f"\nSaved {len(out_dict)} molecules to {output_path}")
    print(f"Skipped {skipped} invalid molecules")
    
    # Compute stats (for z-scoring)
    print("\nComputing statistics...")
    prop_list = ['qed', 'HeavyAtomMolWt', 'MolWt', 'BalabanJ', 'BertzCT', 'Ipc', 
                 'TPSA', 'NumHAcceptors', 'NumHDonors', 'RingCount', 'MolLogP', 'SAscore', 'FSP3']
    stats = {}
    for prop in prop_list:
        values = [out_dict[smi][prop] for smi in out_dict.keys() 
                  if not np.isnan(out_dict[smi][prop])]
        if values:
            stats[prop] = {'mean': np.mean(values), 'std': np.std(values)}
        else:
            stats[prop] = {'mean': 0.0, 'std': 1.0}
    
    stats_path = 'datasets/MOSES_stats.npy'
    np.save(stats_path, stats)
    print(f"Saved statistics to {stats_path}")

