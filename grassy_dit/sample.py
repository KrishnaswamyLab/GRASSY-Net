"""
Sampling script for GRASSY-DiT.
Generates molecules conditioned on scattering moments with optional scaffold constraints.
"""
import argparse
import numpy as np
import torch
from rdkit import Chem
from grassy_dit.train import ScatteringGraphDIT


def smiles_to_scaffold(smiles, max_nodes, atom_decoder, bond_decoder=None):
    """Convert SMILES to one-hot tensors for scaffold constraint."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid scaffold SMILES: {smiles}")
    
    n_atoms = mol.GetNumAtoms()
    
    # Atom types
    X = torch.zeros(max_nodes, len(atom_decoder))
    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol in atom_decoder:
            X[i, atom_decoder.index(symbol)] = 1.0
    
    # Bonds
    num_bond_types = len(bond_decoder) if bond_decoder else 5
    E = torch.zeros(max_nodes, max_nodes, num_bond_types)
    E[:, :, 0] = 1.0  # default no-bond
    
    # Build bond map from decoder if available, else use default
    if bond_decoder:
        bond_map = {}
        for idx, name in enumerate(bond_decoder):
            if name == 'SINGLE': bond_map[Chem.BondType.SINGLE] = idx
            elif name == 'DOUBLE': bond_map[Chem.BondType.DOUBLE] = idx
            elif name == 'TRIPLE': bond_map[Chem.BondType.TRIPLE] = idx
            elif name == 'AROMATIC': bond_map[Chem.BondType.AROMATIC] = idx
    else:
        bond_map = {
            Chem.BondType.SINGLE: 1,
            Chem.BondType.DOUBLE: 2,
            Chem.BondType.TRIPLE: 3,
            Chem.BondType.AROMATIC: 4,
        }
    
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond_map.get(bond.GetBondType(), 0)
        E[i, j, :] = 0
        E[j, i, :] = 0
        E[i, j, bt] = 1.0
        E[j, i, bt] = 1.0
    
    # Node mask for scaffold atoms
    node_mask = torch.zeros(max_nodes, dtype=torch.bool)
    node_mask[:n_atoms] = True
    
    return X, E, node_mask, n_atoms


def main():
    parser = argparse.ArgumentParser(description='Generate molecules with GRASSY-DiT')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--scattering', required=True, help='Path to target scattering .npy file')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples per scattering')
    parser.add_argument('--num_nodes', type=int, default=None, help='Number of atoms (None = sample from training dist)')
    parser.add_argument('--index', type=int, default=None, help='Index of scattering vector (if file has multiple)')
    parser.add_argument('--scaffold', default=None, help='Scaffold SMILES to preserve during generation')
    parser.add_argument('--output', default='generated.txt', help='Output file')
    args = parser.parse_args()
    
    # Load model
    model = ScatteringGraphDIT()
    model.load_from_local(args.checkpoint)
    
    # Load scattering
    scattering = np.load(args.scattering)
    if scattering.ndim == 2:
        idx = args.index if args.index is not None else 0
        scattering = scattering[idx]
    
    # Process scaffold if provided
    scaffold_X, scaffold_E, scaffold_node_mask = None, None, None
    if args.scaffold:
        scaffold_X, scaffold_E, scaffold_node_mask, scaffold_size = smiles_to_scaffold(
            args.scaffold,
            model.max_node,
            model.dataset_info['atom_decoder'],
            model.dataset_info.get('bond_decoder', None)
        )
        # Expand for batch
        scaffold_X = scaffold_X.unsqueeze(0).expand(args.num_samples, -1, -1)
        scaffold_E = scaffold_E.unsqueeze(0).expand(args.num_samples, -1, -1, -1)
        scaffold_node_mask = scaffold_node_mask.unsqueeze(0).expand(args.num_samples, -1)
        
        # If num_nodes not specified, use scaffold size + some extra (should never really be used)
        if args.num_nodes is None:
            args.num_nodes = scaffold_size + 5
            print(f"Scaffold has {scaffold_size} atoms, generating {args.num_nodes} total atoms")
    
    # Generate
    smiles_list = model.generate(
        scattering=scattering,
        num_nodes=args.num_nodes,
        batch_size=args.num_samples,
        scaffold_X=scaffold_X,
        scaffold_E=scaffold_E,
        scaffold_node_mask=scaffold_node_mask,
    )
    
    # Save results
    valid_smiles = [s for s in smiles_list if s is not None]
    print(f"Generated {len(valid_smiles)}/{len(smiles_list)} valid molecules")
    
    with open(args.output, 'w') as f:
        for smi in valid_smiles:
            f.write(smi + '\n')
    
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()

# Examples:
# python -m grassy_dit.sample --checkpoint model.pt --scattering target.npy --num_samples 10
# python -m grassy_dit.sample --checkpoint model.pt --scattering target.npy --scaffold "c1ccccc1" --num_samples 5