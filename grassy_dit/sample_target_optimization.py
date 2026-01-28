"""
Full sampling script for SCULPT with latent space optimization.
Generates molecules by optimizing in SCULPT's latent space and using the optimized
scattering moments to condition molecule generation.

Workflow:
1. Load input molecule SMILES
2. Compute scattering moments from input molecule
3. Encode scattering into SCULPT latent space
4. Optimize in latent space for target property (gradient ascent)
5. Decode optimized latent back to scattering moments
6. Generate molecules conditioned on optimized scattering
"""
import argparse
from glob import glob
from html import parser
import numpy as np
import torch
import yaml
from rdkit import Chem
from typing import Optional, List

import yaml
import os

from grassy_dit.train import ScatteringGraphDIT
from models.GRASSY_model import GRASSY
from models.LatentOptimization import LatentOptimizer
from models.ScatteringTransform import GraphScatteringTransform

from evaluation.utils import (
    smiles_to_pyg_data,
    compute_scattering_from_smiles,
)
from utils.config_utils import config_to_hparams

def load_dit_model(checkpoint_path: str, config_path: str, device: str = "cpu", scattering_path=None):
    """Load GRASSY-DiT model from checkpoint."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = ScatteringGraphDIT(config)
    model.device = torch.device(device)

    # Load checkpoint early so we can infer dimensions if scattering file is missing
    checkpoint = torch.load(checkpoint_path, map_location=device)

    scattering_cfg = config.get('scattering', {})
    J = scattering_cfg.get('J', 4)
    num_levels = 1 + J + J * (J - 1) // 2
    num_moments = scattering_cfg.get('num_moments', 4)

    if scattering_path is not None:
        scattering_data = np.load(scattering_path)
        model.num_atom_types = scattering_data.shape[-1] // (num_levels * num_moments)
    else:
        state = checkpoint.get("model_state_dict", {})
        level_proj_w = state.get("denoiser.scatter_tokenizer.level_proj.weight")
        pos = state.get("denoiser.scatter_tokenizer.pos")

        if level_proj_w is not None:
            model.num_atom_types = level_proj_w.shape[1] // num_moments
        elif pos is not None:
            model.num_atom_types = pos.shape[1] - num_levels

    model.num_levels = num_levels
    model.num_moments = num_moments
    model.J = J

    model._initialize_model(model.model_class, checkpoint)
    model.is_fitted_ = True
    model.fitting_loss = [0.0]
    model.fitting_epoch = 0

    hparams = checkpoint.get('hyperparameters', {})
    if not getattr(model, "dataset_info", None):
        model.dataset_info = hparams.get("dataset_info", None)

    return model

def load_grassy_model(checkpoint_dir, device="cpu"):
    """Load the GRASSY autoencoder model with its config."""
    
    # Load config
    config_path = os.path.join(checkpoint_dir, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint_path = glob(os.path.join(checkpoint_dir, 'best-epoch=*.ckpt'))[0]
    
    # Load checkpoint to extract the saved hparams
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    print("hyper_parameters:", checkpoint['hyper_parameters'])
    print("type:", type(checkpoint['hyper_parameters']))
    # Get hparams from checkpoint (Lightning saves them automatically)
    if 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        model = GRASSY.load_from_checkpoint(checkpoint_path, hparams=hparams, map_location=device)
    else:
        raise ValueError("No hyper_parameters found in checkpoint")
    
    model.eval()
    return model, config

def load_scattering_model(
    in_channels: int = 10,
    J: int = 4,
    num_moments: int = 4,
    device: str = "cpu"
):
    """Load fixed scattering transform model."""
    model = GraphScatteringTransform(
        in_channels=in_channels, J=J, num_moments=num_moments
    )
    model = model.to(device)
    model.eval()
    return model

def load_scattering_vector(scattering_path, index=None):
    scattering = np.load(scattering_path)
    if scattering.ndim == 2:
        idx = index if index is not None else 0
        scattering = scattering[idx]
    return scattering


def build_dummy_scattering(model):
    num_atom_types = model.model.denoiser.scatter_tokenizer.num_atom_types
    num_levels = model.model.denoiser.scatter_tokenizer.num_levels
    num_moments = model.model.denoiser.scatter_tokenizer.num_moments
    scattering_dim = num_atom_types * num_levels * num_moments
    return np.ones(scattering_dim, dtype=np.float32)

def get_atom_types(dit_model) -> List[str]:
    """Extract atom types from DiT model."""
    if hasattr(dit_model, "dataset_info") and dit_model.dataset_info:
        if "atom_decoder" in dit_model.dataset_info:
            return dit_model.dataset_info["atom_decoder"]
    return ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"]


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
    parser.add_argument('--config', default='configs/ZINC/BBAB/BBAB_dit_config.yaml', help='Path to config yaml')
    parser.add_argument('--grassy_checkpoint_dir', required=True, help='Path to GRASSY autoencoder checkpoint')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples per scattering')
    parser.add_argument('--num_nodes', type=int, default=None, help='Number of atoms (None = sample from training dist)')
    parser.add_argument('--index', type=int, default=None, help='Index of scattering vector (if file has multiple)')
    parser.add_argument('--input_molecule', default=None, help='Full molecule SMILES (required when using --scaffold or --remove-atoms)')
    parser.add_argument('--scaffold', default=None, help='Scaffold SMARTS pattern to preserve')
    parser.add_argument('--remove-atoms', default=None, help='Comma-separated atom indices to remove (e.g., "0,1,2")')
    parser.add_argument('--unconditional', action='store_true', help='Ignore scattering and generate unconditionally')
    parser.add_argument('--output', default='generated.txt', help='Output file')
    parser.add_argument('--save_trajectory', action='store_true',help='Save optimization trajectory')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',help='Device to use')
    parser.add_argument('--property_idx', type=int, default=4, help='Index of property to optimize')
    parser.add_argument('--step_size_latent', type=float, default=0.1, help='Step size for latent optimization')
    parser.add_argument('--n_steps_latent', type=int, default=20, help='Number of gradient ascent steps')
    args = parser.parse_args()
    
    device = args.device
    print(f"Using device: {device}")
    # Load DiT model
    print("\nLoading GRASSY-DiT model...")
    dit_model = load_dit_model(args.checkpoint, args.config, device)

    print("Processing Scaffold...")
    # Process scaffold if provided
    scaffold_X, scaffold_E, scaffold_mask = None, None, None
    if args.scaffold or args.remove_atoms:
        if args.input_molecule is None:
            raise ValueError("--input_molecule is required when using --scaffold or --remove-atoms")
        if args.scaffold and args.remove_atoms:
            raise ValueError("Cannot use both --scaffold and --remove-atoms")
        
        # Parse remove_atoms if provided
        remove_indices = None
        if args.remove_atoms:
            remove_indices = [int(x.strip()) for x in args.remove_atoms.split(',')]
        
        scaffold_X, scaffold_E, scaffold_mask, node_mask, n_atoms = smiles_to_scaffold(
            args.input_molecule,
            dit_model.max_node,
            dit_model.dataset_info['atom_decoder'],
            dit_model.dataset_info.get('bond_decoder', None),
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
    
    # Load latent optimizer
    print("Loading GRASSY model...")
    grassy_model,grassy_configs = load_grassy_model(args.grassy_checkpoint_dir, device)

    print("Loading scattering transform...")
    atom_types = get_atom_types(dit_model)
    scattering_model = load_scattering_model(
        in_channels=len(atom_types),
        J=dit_model.J,
        num_moments=dit_model.num_moments,
        device=device
    )

    # Compute scattering from input molecule
    print(f"\nInput molecule: {args.input_molecule}")
    mol = Chem.MolFromSmiles(args.input_molecule)
    if mol is None:
        raise ValueError(f"Invalid input molecule SMILES: {args.input_molecule}")
    print(f"Number of atoms: {mol.GetNumAtoms()}")

    input_scattering = compute_scattering_from_smiles(
        [args.input_molecule],
        scattering_model,
        atom_types,
        device
    ).to(device)

    print(f"Scattering shape: {input_scattering.shape}")

    # Encode into GRASSY latent space
    print("\nEncoding into latent space...")
    with torch.no_grad():
        z, mu, logvar = grassy_model.embed(input_scattering)
    print(f"Latent dimension: {z.shape}")

    # Set up optimizer
    print(f"\nOptimizing for property {args.property_idx} with {args.n_steps_latent} steps...")
    latent_optimizer = LatentOptimizer(
        grassy_model,
        property_idx=args.property_idx,
        step_size=args.step_size_latent
    )
    ## Optimization by gradient ascent in latent space.
    latent_trajectory, decoded_trajectory, num_atoms_trajectory = latent_optimizer.optimize(
        z, n_steps=args.n_steps_latent, return_decoded=True
    )

    print(f"\nOptimization complete!")

    # Save trajectory if requested
    if args.save_trajectory:
        trajectory_file = args.output.replace('.txt', '_trajectory.npz')
        np.savez(
            trajectory_file,
            latent=[lat.cpu().numpy() for lat in latent_trajectory],
            scattering=[scat.cpu().numpy() for scat in decoded_trajectory],
            num_atoms=[na.cpu().numpy() for na in num_atoms_trajectory]
        )
        print(f"Trajectory saved to {trajectory_file}")
    
    # Get final optimized scattering
    scattering_final = decoded_trajectory[-1]
    num_atoms_pred = num_atoms_trajectory[-1]

        
    num_nodes = int(round(num_atoms_pred.item()))
    print(f"Predicted num_atoms after optimization: {num_atoms_pred.item():.1f}")
    print(f"Generating molecules with {num_nodes} atoms")
    
    # Convert scattering to numpy for generation
    scattering_np = scattering_final.detach().cpu().numpy()
    if scattering_np.ndim == 2:
        scattering_np = scattering_np[0]  # Remove batch dimension

    # Generate molecules
    print(f"\nGenerating {args.num_samples} molecules...")
    smiles_list = dit_model.generate(
        scattering=scattering_np,
        num_nodes=num_nodes,
        batch_size=args.num_samples,
        scaffold_X=scaffold_X,
        scaffold_E=scaffold_E,
        scaffold_node_mask=scaffold_mask,
    )
    
    # Save results
    valid_smiles = [s for s in smiles_list if s is not None]
    print(f"Generated {len(valid_smiles)}/{len(smiles_list)} valid molecules")

    with open(args.output, 'w') as f:
        f.write(f"# Input molecule: {args.input_molecule}\n")
        f.write(f"# Property optimized: {args.property_idx}\n")
        f.write(f"# Optimization steps: {args.n_steps_latent}\n")
        f.write(f"# Predicted num_atoms: {num_atoms_pred.item():.1f}\n")
        f.write(f"# Generated num_atoms: {num_nodes}\n")
        f.write("#\n")
        for smi in valid_smiles:
            f.write(smi + '\n')

    print(f"Saved to {args.output}")

    # Print some generated molecules
    if valid_smiles:
        print("\nSample generated molecules:")
        for i, smi in enumerate(valid_smiles[:5]):
            print(f"  {i+1}. {smi}")


if __name__ == "__main__":
    main()

# Examples:
# Basic generation:
# python -m grassy_dit.sample_target_optimization --checkpoint model.pt --grassy_checkpoint grassy_checkpoint.pt --num_samples 10
#
# Scaffold mode (keep benzene ring via SMARTS):
# python -m grassy_dit.sample_target_optimization --checkpoint model.pt --grassy_checkpoint grassy_checkpoint.pt --input_molecule "COc1ccccc1N" --scaffold "c1ccccc1" --num_samples 5 --num_gradient_ascent_steps 5
#
# Removal mode (remove atoms 0,1,2):
# python -m grassy_dit.sample_target_optimization --checkpoint model.pt --grassy_checkpoint grassy_checkpoint.pt --input_molecule "COc1ccccc1N" --remove-atoms "0,1,2" --num_samples 5
#
# Override output size (generate 35 atoms instead of original molecule size):
# python -m grassy_dit.sample_target_optimization --checkpoint model.pt --grassy_checkpoint grassy_checkpoint.pt --input_molecule "COc1ccccc1N" --scaffold "c1ccccc1" --num_nodes 35 --num_samples 5

# python -m grassy_dit.sample_target_optimization \
#     --checkpoint checkpoints/bace/jan_25/checkpoint_best.pt \
#     --grassy_checkpoint_dir  outputs/BACE_fixed_regress_nokld_2026-01-26-15-38-28/ \
#     --input_molecule "COc1ccccc1N" \
#     --scaffold "c1ccccc1" \
#     --property_idx 0 \
#     --n_steps_latent 10 \
#     --num_samples 5
