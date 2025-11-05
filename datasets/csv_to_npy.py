"""Convert a CSV with SMILES into a .npy tranche file compatible with the repo.

Usage:
    python csv_to_npy.py --input molecules.csv --smiles_col smiles --output BBAB_subset.npy --ki_col Ki

The script will compute common descriptors with RDKit and save a dict mapping
SMILES -> property-dict (same shape as existing *_subset.npy files in this repo).
"""

import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, QED
except Exception as e:
    raise ImportError("RDKit must be installed to compute molecular descriptors: %s" % e)


PROP_KEYS = ['qed', 'HeavyAtomMolWt', 'MolWt', 'BalabanJ', 'BertzCT', 'Ipc',
             'TPSA', 'NumHAcceptors', 'NumHDonors', 'RingCount']


def compute_props(mol):
    """Compute the property dict for a single RDKit Mol object.
    Returns dict with keys in PROP_KEYS (values are floats).
    If a descriptor fails, set numpy.nan.
    """
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
        # estimate heavy atom molecular weight as MolWt - H_count * 1.00794
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
        # fallback to Descriptors.NumHAcceptors if available
        try:
            out['NumHAcceptors'] = float(Descriptors.NumHAcceptors(mol))
        except Exception:
            out['NumHAcceptors'] = float('nan')

    try:
        out['NumHDonors'] = float(rdMolDescriptors.CalcNumHBD(mol))
    except Exception:
        try:
            out['NumHDonors'] = float(Descriptors.NumHDonors(mol))
        except Exception:
            out['NumHDonors'] = float('nan')

    try:
        out['RingCount'] = float(mol.GetRingInfo().NumRings())
    except Exception:
        out['RingCount'] = float('nan')

    return out


def csv_to_npy(input_csv, smiles_col='smiles', output_path='tranche_subset.npy', ki_col=None, smiles_header_case_insensitive=True):
    df = pd.read_csv(input_csv)
    for smi in df["SMILES"].head(10):
        mol = Chem.MolFromSmiles(str(smi))
        print(smi, "=>", mol.GetNumAtoms(), "atoms,", mol.GetNumBonds(), "bonds")
    # import pdb; pdb.set_trace()
    # try case-insensitive lookup for smiles column
    if smiles_header_case_insensitive:
        cols_lower = {c.lower(): c for c in df.columns}
        if smiles_col not in df.columns and smiles_col.lower() in cols_lower:
            smiles_col = cols_lower[smiles_col.lower()]
        if ki_col and ki_col not in df.columns and ki_col.lower() in cols_lower:
            ki_col = cols_lower[ki_col.lower()]

    out_dict = {}
    skipped = 0
    for idx, row in df.iterrows():
        smi = row.get(smiles_col)
        if pd.isna(smi):
            skipped += 1
            continue
        smi = str(smi).strip()
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                warnings.warn(f"RDKit failed to parse SMILES: {smi} (skipping)")
                skipped += 1
                continue
        except Exception:
            warnings.warn(f"Exception parsing SMILES: {smi} (skipping)")
            skipped += 1
            continue

        props = compute_props(mol)

        # include Ki if present
        if ki_col and ki_col in row:
            try:
                props['Ki'] = float(row[ki_col]) if not pd.isna(row[ki_col]) else float('nan')
            except Exception:
                props['Ki'] = float('nan')

        out_dict[smi] = props

    # save as numpy .npy containing a dict (like existing tranche files)
    np.save(output_path, out_dict)
    return output_path, skipped, len(out_dict)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True, help='Input CSV file containing SMILES')
    parser.add_argument('--smiles_col', default='smiles', help='Column name with SMILES strings')
    parser.add_argument('--ki_col', default=None, help='Optional Ki column name')
    parser.add_argument('--output', '-o', default='fields_1.npy', help='Output .npy path')
    args = parser.parse_args()

    out, skipped, kept = csv_to_npy(args.input, smiles_col=args.smiles_col, output_path=args.output, ki_col=args.ki_col)
    print(f"Wrote {out}: kept={kept}, skipped={skipped}")
