
import os
import sys
from rdkit.Chem import Descriptors, rdMolDescriptors, QED

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
    

PROPERTIES_TO_COMPUTE = [
    'qed',
    'HeavyAtomMolWt',
    'MolWt',
    'TPSA',
    'MolLogP',
    'SAS',
    'BalabanJ',
    'BertzCT',
    'FSP3'
]

PROPERTY_REGISTRY = {
    'qed': (lambda mol: QED.qed(mol), "Quantitative Estimate of Drug-likeness"),
    'HeavyAtomMolWt': (lambda mol: Descriptors.HeavyAtomMolWt(mol), "Heavy Atom Molecular Weight"),
    'MolWt': (lambda mol: Descriptors.MolWt(mol), "Molecular Weight"),
    'TPSA': (lambda mol: rdMolDescriptors.CalcTPSA(mol), "Topological Polar Surface Area"),
    'MolLogP': (lambda mol: Descriptors.MolLogP(mol), "LogP"),
    'SAS': (compute_sas, "Synthetic Accessibility Score"),
    'SCS': (compute_scs, "Synthetic Complexity Score"),
    
    # From your second list (commented out ones)
    'BalabanJ': (lambda mol: Descriptors.BalabanJ(mol), "Balaban J index"),
    'BertzCT': (lambda mol: Descriptors.BertzCT(mol), "Bertz complexity index"),
    'Ipc': (lambda mol: Descriptors.Ipc(mol), "Information content"),
    'FSP3': (lambda mol: rdMolDescriptors.CalcFractionCSP3(mol), "Fraction SP3 carbons"),
}