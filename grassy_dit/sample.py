"""
Sampling script for GRASSY-DiT.
Generates molecules conditioned on scattering moments with optional scaffold constraints.
"""
import argparse
import numpy as np
import torch
from rdkit import Chem
from grassy_dit.train import ScatteringGraphDIT


def smiles_to_scaffold(full_smiles, max_nodes, atom_decoder, bond_decoder=None,
                       scaffold_pattern=None, remove_indices=None, num_nodes=None):
    """
    Build graph from full molecule with scaffold mask indicating fixed atoms.
    
    Args:
        full_smiles: Complete molecule SMILES
        max_nodes: Maximum nodes in graph
        atom_decoder: List of atom symbols
        bond_decoder: List of bond type names (optional)
        scaffold_pattern: SMARTS pattern to KEEP - atoms matching this are fixed
        remove_indices: List of atom indices to REMOVE - atoms NOT in this list are fixed
        num_nodes: Desired output size (defaults to molecule's atom count)
        
    Exactly one of scaffold_pattern or remove_indices must be provided.
    
    Returns:
        X: [max_nodes, num_atom_types] atom features
        E: [max_nodes, max_nodes, num_bond_types] bond features
        scaffold_mask: [max_nodes] bool - True = fixed scaffold atom
        node_mask: [max_nodes] bool - True = real atom, False = padding
        n_atoms: number of atoms in full molecule
    """
    if (scaffold_pattern is None) == (remove_indices is None):
        raise ValueError("Exactly one of scaffold_pattern or remove_indices must be provided")
    
    full_mol = Chem.MolFromSmiles(full_smiles)
    if full_mol is None:
        raise ValueError(f"Invalid full molecule SMILES: {full_smiles}")
    
    n_atoms = full_mol.GetNumAtoms()
    
    # Determine which atoms are scaffold (fixed)
    if scaffold_pattern is not None:
        # Mode 1: SMARTS pattern matching - keep atoms that match
        scaffold_mol = Chem.MolFromSmarts(scaffold_pattern)
        if scaffold_mol is None:
            raise ValueError(f"Invalid SMARTS pattern: {scaffold_pattern}")
        
        match = full_mol.GetSubstructMatch(scaffold_mol)
        if not match:
            raise ValueError(f"Scaffold pattern not found in molecule")
        scaffold_indices = set(match)
    else:
        # Mode 2: Removal - keep atoms NOT in remove list
        scaffold_indices = set(range(n_atoms)) - set(remove_indices)
    
    # Build atom features from full molecule
    X = torch.zeros(max_nodes, len(atom_decoder))
    for i, atom in enumerate(full_mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol in atom_decoder:
            X[i, atom_decoder.index(symbol)] = 1.0
    
    # Build bond features from full molecule
    num_bond_types = len(bond_decoder) if bond_decoder else 5
    E = torch.zeros(max_nodes, max_nodes, num_bond_types)
    E[:, :, 0] = 1.0  # default no-bond
    
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
    
    for bond in full_mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond_map.get(bond.GetBondType(), 0)
        E[i, j, :] = 0
        E[j, i, :] = 0
        E[i, j, bt] = 1.0
        E[j, i, bt] = 1.0
    
    # Node mask: real atoms vs padding
    # Use num_nodes if specified, otherwise use molecule's atom count
    effective_nodes = num_nodes if num_nodes is not None else n_atoms
    node_mask = torch.zeros(max_nodes, dtype=torch.bool)
    node_mask[:effective_nodes] = True
    
    # Scaffold mask: which atoms are fixed
    scaffold_mask = torch.zeros(max_nodes, dtype=torch.bool)
    for idx in scaffold_indices:
        scaffold_mask[idx] = True

    # Zero out non-scaffold atoms (remove them from graph representation)
    X[~scaffold_mask] = 0
    E[~scaffold_mask, :, :] = 0
    E[:, ~scaffold_mask, :] = 0
    
    return X, E, scaffold_mask, node_mask, n_atoms

def main():
    parser = argparse.ArgumentParser(description='Generate molecules with GRASSY-DiT')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--scattering', required=True, help='Path to target scattering .npy file')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples per scattering')
    parser.add_argument('--num_nodes', type=int, default=None, help='Number of atoms (None = sample from training dist)')
    parser.add_argument('--index', type=int, default=None, help='Index of scattering vector (if file has multiple)')
    parser.add_argument('--molecule', default=None, help='Full molecule SMILES (required when using --scaffold or --remove-atoms)')
    parser.add_argument('--scaffold', default=None, help='Scaffold SMARTS pattern to preserve')
    parser.add_argument('--remove-atoms', default=None, help='Comma-separated atom indices to remove (e.g., "0,1,2")')
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
    scaffold_X, scaffold_E, scaffold_mask = None, None, None
    if args.scaffold or args.remove_atoms:
        if args.molecule is None:
            raise ValueError("--molecule is required when using --scaffold or --remove-atoms")
        if args.scaffold and args.remove_atoms:
            raise ValueError("Cannot use both --scaffold and --remove-atoms")
        
        # Parse remove_atoms if provided
        remove_indices = None
        if args.remove_atoms:
            remove_indices = [int(x.strip()) for x in args.remove_atoms.split(',')]
        
        scaffold_X, scaffold_E, scaffold_mask, node_mask, n_atoms = smiles_to_scaffold(
            args.molecule,
            model.max_node,
            model.dataset_info['atom_decoder'],
            model.dataset_info.get('bond_decoder', None),
            scaffold_pattern=args.scaffold,
            remove_indices=remove_indices,
            num_nodes=args.num_nodes,
        )
        
        scaffold_count = scaffold_mask.sum().item()
        print(f"Full molecule: {n_atoms} atoms")
        print(f"Scaffold (fixed): {scaffold_count} atoms")
        print(f"To regenerate: {n_atoms - scaffold_count} atoms")
        
        # Expand for batch
        scaffold_X = scaffold_X.unsqueeze(0).expand(args.num_samples, -1, -1)
        scaffold_E = scaffold_E.unsqueeze(0).expand(args.num_samples, -1, -1, -1)
        scaffold_mask = scaffold_mask.unsqueeze(0).expand(args.num_samples, -1)
        
        # Default to full molecule size, but allow override via --num_nodes
        if args.num_nodes is None:
            args.num_nodes = n_atoms
            print(f"Output size: {args.num_nodes} atoms (from molecule)")
        else:
            print(f"Output size: {args.num_nodes} atoms (specified)")
    
    # Generate
    smiles_list = model.generate(
        scattering=scattering,
        num_nodes=args.num_nodes,
        batch_size=args.num_samples,
        scaffold_X=scaffold_X,
        scaffold_E=scaffold_E,
        scaffold_node_mask=scaffold_mask,
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
# Basic generation:
# python -m grassy_dit.sample --checkpoint model.pt --scattering target.npy --num_samples 10
#
# Scaffold mode (keep benzene ring via SMARTS):
# python -m grassy_dit.sample --checkpoint model.pt --scattering target.npy --molecule "COc1ccccc1N" --scaffold "c1ccccc1" --num_samples 5
#
# Removal mode (remove atoms 0,1,2):
# python -m grassy_dit.sample --checkpoint model.pt --scattering target.npy --molecule "COc1ccccc1N" --remove-atoms "0,1,2" --num_samples 5
#
# Override output size (generate 35 atoms instead of original molecule size):
# python -m grassy_dit.sample --checkpoint model.pt --scattering target.npy --molecule "COc1ccccc1N" --scaffold "c1ccccc1" --num_nodes 35 --num_samples 5