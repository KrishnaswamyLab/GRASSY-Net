"""Download ZINC 12K subset and convert to GRASSY format.

Usage:
    python datasets/prepare_zinc12k.py
"""

import numpy as np
from torch_geometric.datasets import ZINC
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, QED
from tqdm import tqdm
import sys
sys.path.insert(0, 'datasets')
import sascorer

def compute_props(mol):
    """Compute properties for a molecule (same as csv_to_npy.py)."""
    out = {}
    try:
        out['qed'] = float(QED.qed(mol))
    except Exception:
        out['qed'] = float('nan')

    try:
        mw = float(Descriptors.MolWt(mol))
        out['MolWt'] = mw
    except Exception:
        out['MolWt'] = float('nan')

    try:
        num_H = sum([atom.GetTotalNumHs() for atom in mol.GetAtoms()])
        out['HeavyAtomMolWt'] = mw - num_H * 1.00794 if not np.isnan(mw) else float('nan')
    except Exception:
        out['HeavyAtomMolWt'] = float('nan')

    try:
        out['BalabanJ'] = float(Descriptors.BalabanJ(mol))
    except Exception:
        out['BalabanJ'] = float('nan')

    try:
        out['BertzCT'] = float(Descriptors.BertzCT(mol))
    except Exception:
        out['BertzCT'] = float('nan')

    try:
        out['Ipc'] = float(Descriptors.Ipc(mol))
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
        out['SAscore'] = float(sascorer.calculateScore(mol))
    except Exception:
        out['SAscore'] = float('nan')

    return out


if __name__ == '__main__':
    print("Downloading ZINC 12K subset...")
    train_data = ZINC(root='data/ZINC', subset=True, split='train')
    val_data = ZINC(root='data/ZINC', subset=True, split='val')
    test_data = ZINC(root='data/ZINC', subset=True, split='test')

    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Extract SMILES from molecular graphs
    print("Reconstructing SMILES from graphs...")
    smiles_list = []
    
    # Atom decoder: PyG ZINC encodes atoms with implicit H's and charges
    # We map to element symbols; RDKit will handle implicit hydrogens
    # Atom decoder with charges: (element, num_explicit_H, formal_charge)
    atom_map = {
        0: ('C', 0, 0), 1: ('O', 0, 0), 2: ('N', 0, 0), 3: ('F', 0, 0),
        4: ('C', 1, 0), 5: ('S', 0, 0), 6: ('Cl', 0, 0), 7: ('O', 0, -1),
        8: ('N', 1, 1), 9: ('Br', 0, 0), 10: ('N', 3, 1), 11: ('N', 2, 1),
        12: ('N', 0, 1), 13: ('N', 0, -1), 14: ('S', 0, -1), 15: ('I', 0, 0),
        16: ('P', 0, 0), 17: ('O', 1, 1), 18: ('N', 1, -1), 19: ('O', 0, 1),
        20: ('S', 0, 1), 21: ('P', 1, 0), 22: ('P', 2, 0), 23: ('C', 2, -1),
        24: ('P', 0, 1), 25: ('S', 1, 1), 26: ('C', 1, -1), 27: ('P', 1, 1),
    }
    
    element_to_atomic = {
        'C': 6, 'N': 7, 'O': 8, 'F': 9, 'P': 15, 
        'S': 16, 'Cl': 17, 'Br': 35, 'I': 53
    }
    
    # Bond decoder: 1=SINGLE, 2=DOUBLE, 3=TRIPLE (note: starts at 1, not 0!)
    bond_map = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE
    }
    
    for dataset_name, dataset in [('train', train_data), ('val', val_data), ('test', test_data)]:
        print(f"Processing {dataset_name}...")
        for data in tqdm(dataset):
            mol = Chem.RWMol()
            
            # Add atoms (data.x is [num_atoms, 1], contains integer indices)
            for i in range(data.num_nodes):
                atom_type_idx = data.x[i, 0].item()
                element, num_h, charge = atom_map.get(atom_type_idx, ('C', 0, 0))
                atomic_num = element_to_atomic[element]
                atom = Chem.Atom(atomic_num)
                atom.SetFormalCharge(charge)
                if num_h > 0:
                    atom.SetNumExplicitHs(num_h)
                mol.AddAtom(atom)
            
            # Add bonds (edge_index is bidirectional, edge_attr is 1D integer array)
            added_bonds = set()
            for i in range(data.edge_index.size(1)):
                src = data.edge_index[0, i].item()
                dst = data.edge_index[1, i].item()
                
                # Only add each bond once (skip reverse direction)
                if src < dst and (src, dst) not in added_bonds:
                    bond_type_idx = data.edge_attr[i].item()
                    bond_type = bond_map.get(bond_type_idx, Chem.BondType.SINGLE)
                    mol.AddBond(src, dst, bond_type)
                    added_bonds.add((src, dst))
            
            try:
                smi = Chem.MolToSmiles(mol)
                smiles_list.append(smi)
            except Exception as e:
                print(f"Failed to generate SMILES: {e}")
                continue

    print(f"Total molecules reconstructed: {len(smiles_list)}")

    # Compute properties
    print("Computing properties...")
    out_dict = {}
    skipped = 0
    
    for smi in tqdm(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"Warning: Failed to parse SMILES: {smi}")
                skipped += 1
                continue
            out_dict[smi] = compute_props(mol)
        except Exception as e:
            print(f"Warning: Exception for {smi}: {e}")
            skipped += 1
            continue

    # Save
    output_path = 'datasets/ZINC12K.npy'
    np.save(output_path, out_dict)
    print(f"\nSaved {len(out_dict)} molecules to {output_path}")
    print(f"Skipped {skipped} invalid molecules")
    
    # Sanity check
    if len(out_dict) > 0:
        sample_smi = list(out_dict.keys())[0]
        print(f"\nSample molecule:")
        print(f"  SMILES: {sample_smi}")
        print(f"  Properties: {out_dict[sample_smi]}")