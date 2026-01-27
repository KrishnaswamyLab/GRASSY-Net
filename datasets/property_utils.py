
import os
import sys
from rdkit.Chem import Descriptors, rdMolDescriptors, QED

def compute_props(mol, bace_label=None):
    """Compute properties for a molecule based on PROPERTIES_TO_COMPUTE list."""
    out = {}
    
    for prop_name in PROPERTIES_TO_COMPUTE:
        if prop_name not in PROPERTY_REGISTRY:
            print(f"Warning: Unknown property '{prop_name}'")
            continue
        compute_fn, _ = PROPERTY_REGISTRY[prop_name]
        try:
            out[prop_name] = float(compute_fn(mol))
        except Exception:
            out[prop_name] = float('nan')
    
    # Special case: BACE activity label (always include if provided)
    if bace_label is not None:
        out['bace_activity'] = float(bace_label)
    
    return out

def compute_sas(mol):
    """Compute Synthetic Accessibility Score (1-10, lower is easier)."""
    try:
        from rdkit.Contrib.SA_Score import sascorer
        return float(sascorer.calculateScore(mol))
    except Exception:
        try:
            # Alternative location in some RDKit versions
            from rdkit.Chem import RDConfig
            sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
            import sascorer
            return float(sascorer.calculateScore(mol))
        except Exception:
            return float('nan')


def compute_scs(mol):
    """Compute Synthetic Complexity Score (1-5, lower is easier)."""
    try:
        # SCScore approximation using BertzCT
        bertz = rdMolDescriptors.CalcBertzCT(mol)
        # Normalize to 1-5 range
        scs = 1 + 4 * min(bertz / 2000, 1.0)
        return float(scs)
    except Exception:
        return float('nan')


def compute_num_atoms(mol):
    """Compute number of heavy atoms in the molecule."""
    return float(mol.GetNumHeavyAtoms())
    
    

PROPERTIES_TO_COMPUTE = [
    'qed',
    'HeavyAtomMolWt',
    'MolWt',
    'TPSA',
    'MolLogP',
    'num_atoms'
    #BELOW PROPERTIES ARE NOT CONSIDERED FOR MOSES
    # 'SAS',
    # 'BalabanJ',
    # 'BertzCT',
    # 'FSP3'
]

PROPERTY_REGISTRY = {
    'qed': (lambda mol: QED.qed(mol), "Quantitative Estimate of Drug-likeness"),
    'HeavyAtomMolWt': (lambda mol: Descriptors.HeavyAtomMolWt(mol), "Heavy Atom Molecular Weight"),
    'MolWt': (lambda mol: Descriptors.MolWt(mol), "Molecular Weight"),
    'TPSA': (lambda mol: rdMolDescriptors.CalcTPSA(mol), "Topological Polar Surface Area"),
    'MolLogP': (lambda mol: Descriptors.MolLogP(mol), "LogP"),
    'num_atoms': (lambda mol: float(mol.GetNumHeavyAtoms()), "Number of Heavy Atoms"),
    'SAS': (compute_sas, "Synthetic Accessibility Score"),
    'SCS': (compute_scs, "Synthetic Complexity Score"),
    
    # From your second list (commented out ones)
    'BalabanJ': (lambda mol: Descriptors.BalabanJ(mol), "Balaban J index"),
    'BertzCT': (lambda mol: Descriptors.BertzCT(mol), "Bertz complexity index"),
    'Ipc': (lambda mol: Descriptors.Ipc(mol), "Information content"),
    'FSP3': (lambda mol: rdMolDescriptors.CalcFractionCSP3(mol), "Fraction SP3 carbons"),
}