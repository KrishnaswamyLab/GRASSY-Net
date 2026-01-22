"""
Shared utilities for MOSES benchmark evaluation.

Provides functions to load models, compute scattering, and convert molecules.
"""

import os
import sys
import torch
import yaml
import numpy as np
from typing import Optional, List, Union
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_dit_model(checkpoint_path: str, device: str = "cuda", config_path: str = None):
    """
    Load a trained GRASSY-DiT model from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint (.pt file)
        device: Device to load the model on ('cuda' or 'cpu')
        config_path: Path to config yaml (defaults to grassy_dit/grassy_dit_config.yaml)

    Returns:
        ScatteringGraphDIT model ready for generation
    """
    from grassy_dit.train import ScatteringGraphDIT

    # Load config
    if config_path is None:
        config_path = PROJECT_ROOT / "grassy_dit" / "grassy_dit_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = ScatteringGraphDIT(config)

    # Load checkpoint (Joao's format doesn't have 'model_name')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model._initialize_model(model.model_class, checkpoint)
    model.is_fitted_ = True
    model.fitting_loss = [0.0]
    model.fitting_epoch = 0

    # Set dataset_info from checkpoint hyperparameters
    hparams = checkpoint.get('hyperparameters', {})
    if not getattr(model, "dataset_info", None):
        model.dataset_info = hparams.get("dataset_info", None)

    model.device = torch.device(device)
    return model


def load_grassy_vae(
    checkpoint_path: str, device: str = "cuda"
) -> "torch.nn.Module":
    """
    Load a trained GRASSY VAE model from checkpoint.

    Args:
        checkpoint_path: Path to the GRASSY VAE checkpoint (.pt or .ckpt file)
        device: Device to load the model on

    Returns:
        GRASSY VAE model ready for encoding/decoding
    """
    from models.GRASSY_model import GRASSY

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        # PyTorch Lightning checkpoint
        hparams = checkpoint.get("hyper_parameters", checkpoint.get("hparams", {}))
        model = GRASSY(hparams)
        model.load_state_dict(checkpoint["state_dict"])
    elif "hparams" in checkpoint:
        # Custom checkpoint with hparams
        model = GRASSY(checkpoint["hparams"])
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Try to infer from state dict keys
        raise ValueError(
            f"Cannot load GRASSY VAE from checkpoint format. "
            f"Keys found: {checkpoint.keys()}"
        )

    model = model.to(device)
    model.eval()
    return model


def load_scattering_model(
    checkpoint_path: Optional[str] = None,
    in_channels: int = 10,
    J: int = 4,
    num_moments: int = 4,
    device: str = "cuda",
):
    """
    Load a scattering model (fixed or learnable).

    Args:
        checkpoint_path: Path to learnable scattering checkpoint (None = use fixed)
        in_channels: Number of atom types (for fixed scattering)
        J: Number of wavelet scales
        num_moments: Number of statistical moments
        device: Device to load model on

    Returns:
        Scattering model (GraphScatteringTransform or MLP_Scattering)
    """
    if checkpoint_path is None:
        # Use fixed (non-learnable) scattering transform
        from models.ScatteringTransform import GraphScatteringTransform

        model = GraphScatteringTransform(
            in_channels=in_channels, J=J, num_moments=num_moments
        )
    else:
        # Load learnable scattering model
        from models.MLP_Scattering_module import MLP_Scattering

        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Initialize from checkpoint hyperparameters
        if "hparams" in checkpoint:
            model = MLP_Scattering(**checkpoint["hparams"])
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            raise ValueError(
                f"Cannot load learnable scattering from checkpoint format"
            )

    model = model.to(device)
    model.eval()
    return model


def smiles_to_pyg_data(smiles: str, atom_types: Optional[List[str]] = None):
    """
    Convert a SMILES string to PyTorch Geometric Data object.

    Args:
        smiles: SMILES string
        atom_types: List of atom symbols to use (default: common organic atoms)

    Returns:
        PyG Data object with x, edge_index, batch attributes
    """
    from rdkit import Chem
    from torch_geometric.data import Data

    if atom_types is None:
        atom_types = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # Build atom features (one-hot encoding)
    num_atoms = mol.GetNumAtoms()
    x = torch.zeros(num_atoms, len(atom_types))

    for i, atom in enumerate(mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol in atom_types:
            x[i, atom_types.index(symbol)] = 1.0
        # Unknown atoms get zero vector (will still contribute to scattering)

    # Build edge index
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])  # Undirected

    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(num_atoms, dtype=torch.long)

    return data


def compute_scattering_from_smiles(
    smiles_list: List[str],
    scattering_model,
    atom_types: Optional[List[str]] = None,
    device: str = "cuda",
    batch_size: int = 64,
) -> torch.Tensor:
    """
    Compute scattering moments for a list of SMILES.

    Args:
        smiles_list: List of SMILES strings
        scattering_model: Scattering transform model
        atom_types: List of atom symbols
        device: Device for computation
        batch_size: Batch size for processing

    Returns:
        Tensor of scattering moments [N, scattering_dim]
    """
    from torch_geometric.data import Batch

    all_scattering = []
    valid_indices = []

    for batch_start in range(0, len(smiles_list), batch_size):
        batch_end = min(batch_start + batch_size, len(smiles_list))
        batch_smiles = smiles_list[batch_start:batch_end]

        # Convert to PyG data
        data_list = []
        for i, smi in enumerate(batch_smiles):
            data = smiles_to_pyg_data(smi, atom_types)
            if data is not None:
                data_list.append(data)
                valid_indices.append(batch_start + i)

        if len(data_list) == 0:
            continue

        # Batch and compute scattering
        batch = Batch.from_data_list(data_list).to(device)

        with torch.no_grad():
            scattering = scattering_model(batch)

        all_scattering.append(scattering.cpu())

    if len(all_scattering) == 0:
        return torch.empty(0, scattering_model.out_shape())

    return torch.cat(all_scattering, dim=0)


def get_conditioning(
    mode: str,
    source_smiles: Optional[List[str]],
    scattering_model,
    grassy_vae=None,
    num_samples: int = 10000,
    device: str = "cuda",
    atom_types: Optional[List[str]] = None,
) -> Optional[torch.Tensor]:
    """
    Get conditioning vectors based on the generation mode.

    Args:
        mode: One of 'unconditional', 'direct', 'full-pipeline', 'latent-sample'
        source_smiles: Source molecules for conditioning (not needed for unconditional/latent-sample)
        scattering_model: Scattering transform model
        grassy_vae: GRASSY VAE model (required for full-pipeline and latent-sample)
        num_samples: Number of samples to generate conditioning for
        device: Device for computation
        atom_types: List of atom symbols

    Returns:
        Conditioning tensor [num_samples, conditioning_dim] or None for unconditional
    """
    if mode == "unconditional":
        return None

    elif mode == "direct":
        # Compute scattering from source molecules
        if source_smiles is None:
            raise ValueError("source_smiles required for 'direct' mode")
        
        # Sample if we have more source molecules than needed
        if len(source_smiles) > num_samples:
            indices = np.random.choice(len(source_smiles), num_samples, replace=False)
            source_smiles = [source_smiles[i] for i in indices]
        elif len(source_smiles) < num_samples:
            # Repeat with replacement
            indices = np.random.choice(len(source_smiles), num_samples, replace=True)
            source_smiles = [source_smiles[i] for i in indices]

        return compute_scattering_from_smiles(
            source_smiles, scattering_model, atom_types, device
        )

    elif mode == "full-pipeline":
        # Scattering → GRASSY encode → decode
        if source_smiles is None:
            raise ValueError("source_smiles required for 'full-pipeline' mode")
        if grassy_vae is None:
            raise ValueError("grassy_vae required for 'full-pipeline' mode")

        # Sample source molecules
        if len(source_smiles) > num_samples:
            indices = np.random.choice(len(source_smiles), num_samples, replace=False)
            source_smiles = [source_smiles[i] for i in indices]
        elif len(source_smiles) < num_samples:
            indices = np.random.choice(len(source_smiles), num_samples, replace=True)
            source_smiles = [source_smiles[i] for i in indices]

        # Compute scattering
        scattering = compute_scattering_from_smiles(
            source_smiles, scattering_model, atom_types, device
        ).to(device)

        # Encode through GRASSY VAE and decode
        with torch.no_grad():
            z, _, _ = grassy_vae.embed(scattering)
            conditioning = grassy_vae.decode(z)

        return conditioning.cpu()

    elif mode == "latent-sample":
        # Sample random z → decode to moments
        if grassy_vae is None:
            raise ValueError("grassy_vae required for 'latent-sample' mode")

        bottle_dim = grassy_vae.bottle_dim

        with torch.no_grad():
            z = torch.randn(num_samples, bottle_dim, device=device)
            conditioning = grassy_vae.decode(z)

        return conditioning.cpu()

    else:
        raise ValueError(f"Unknown mode: {mode}. Expected one of: "
                        "unconditional, direct, full-pipeline, latent-sample")


def get_atom_types_from_model(dit_model) -> List[str]:
    """
    Extract atom types from a loaded DiT model's dataset info.

    Args:
        dit_model: Loaded ScatteringGraphDIT model

    Returns:
        List of atom type symbols
    """
    if hasattr(dit_model, "dataset_info") and "atom_decoder" in dit_model.dataset_info:
        return dit_model.dataset_info["atom_decoder"]

    # Default fallback
    return ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"]


def format_metrics_table(
    metrics_dict: dict, model_names: Optional[List[str]] = None
) -> str:
    """
    Format metrics as a markdown table for comparison.

    Args:
        metrics_dict: Dict mapping model names to their metrics dicts
        model_names: Order of models to display (optional)

    Returns:
        Markdown-formatted table string
    """
    if model_names is None:
        model_names = list(metrics_dict.keys())

    # Collect all metric names
    all_metrics = set()
    for metrics in metrics_dict.values():
        all_metrics.update(metrics.keys())

    # Sort metrics in a sensible order
    priority_metrics = [
        "valid",
        "unique@1k",
        "unique@10k",
        "FCD/Test",
        "SNN/Test",
        "Frag/Test",
        "Scaf/Test",
        "IntDiv",
        "IntDiv2",
        "Filters",
        "Novelty",
    ]

    sorted_metrics = []
    for m in priority_metrics:
        if m in all_metrics:
            sorted_metrics.append(m)
            all_metrics.discard(m)
    sorted_metrics.extend(sorted(all_metrics))

    # Build table
    header = "| Metric | " + " | ".join(model_names) + " |"
    separator = "|" + "|".join(["---"] * (len(model_names) + 1)) + "|"

    rows = [header, separator]
    for metric in sorted_metrics:
        row = f"| {metric} |"
        for model in model_names:
            value = metrics_dict.get(model, {}).get(metric, "-")
            if isinstance(value, float):
                row += f" {value:.4f} |"
            else:
                row += f" {value} |"
        rows.append(row)

    return "\n".join(rows)
