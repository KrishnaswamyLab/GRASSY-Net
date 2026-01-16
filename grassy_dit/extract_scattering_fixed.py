"""
Extract scattering coefficients using fixed (non-learnable) GraphScatteringTransform.
Outputs: scattering_moments.npy and molecules.csv for DiT training.

Unlike the original extract_scattering.py, this version does NOT require a
pre-trained scattering model - it uses the deterministic GraphScatteringTransform.

Usage:
    python grassy_dit/extract_scattering_fixed.py
    python grassy_dit/extract_scattering_fixed.py --dataset datasets/ZINC12K.npy --output grassy_dit/data/
    python grassy_dit/extract_scattering_fixed.py --J 4 --moments 4
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.loader import DataLoader

from models.ScatteringTransform import GraphScatteringTransform
from datasets.load_ZINC_tranche import ZINCDataset


def extract_scattering(
    dataset_path: str,
    output_dir: str,
    stats_path: str = None,
    J: int = 4,
    num_moments: int = 4,
    batch_size: int = 64,
    device: str = "auto",
):
    """
    Extract scattering coefficients for all molecules in a dataset.

    Args:
        dataset_path: Path to dataset .npy file
        output_dir: Directory to save outputs
        stats_path: Optional path to property statistics file
        J: Number of wavelet scales
        num_moments: Number of statistical moments
        batch_size: Batch size for processing
        device: Device to use ('auto', 'cuda', 'mps', 'cpu')
    """
    # Setup device
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = ZINCDataset(dataset_path, prop_stat_dict=stats_path, include_ki=False)
    print(f"Loaded {len(dataset)} molecules")
    print(f"Node features: {dataset.num_node_features}")

    # Create fixed scattering transform
    scattering = GraphScatteringTransform(
        in_channels=dataset.num_node_features,
        J=J,
        num_moments=num_moments,
    ).to(device)
    scattering.eval()

    print(f"\nScattering configuration:")
    print(f"  - Wavelet scales (J): {J}")
    print(f"  - Moments: {num_moments}")
    print(f"  - Output dimension: {scattering.out_shape()}")

    # Create dataloader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Extract scattering coefficients
    print(f"\nExtracting scattering coefficients...")
    all_moments = []

    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(device)
            moments = scattering(batch)
            all_moments.append(moments.cpu().numpy())

    # Concatenate all batches
    scattering_coeffs = np.vstack(all_moments)
    print(f"\nScattering coefficients shape: {scattering_coeffs.shape}")

    # Get SMILES list (preserving order)
    all_smiles = dataset.smi
    print(f"SMILES count: {len(all_smiles)}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save outputs
    scattering_path = os.path.join(output_dir, "scattering_moments.npy")
    molecules_path = os.path.join(output_dir, "molecules.csv")

    np.save(scattering_path, scattering_coeffs)
    pd.DataFrame({"smiles": all_smiles}).to_csv(molecules_path, index=False)

    print(f"\nSaved to {output_dir}:")
    print(f"  - {scattering_path}")
    print(f"  - {molecules_path}")

    # Check for NaN values
    nan_count = np.isnan(scattering_coeffs).sum()
    if nan_count > 0:
        print(f"\nWarning: {nan_count} NaN values found in scattering coefficients")

    return scattering_coeffs, all_smiles


def main():
    parser = argparse.ArgumentParser(
        description="Extract scattering coefficients using fixed GraphScatteringTransform"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/ZINC12K.npy",
        help="Path to dataset .npy file",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="datasets/ZINC12K_stats.npy",
        help="Path to property statistics file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="grassy_dit/data/",
        help="Output directory",
    )
    parser.add_argument(
        "--J",
        type=int,
        default=4,
        help="Number of wavelet scales (default: 4)",
    )
    parser.add_argument(
        "--moments",
        type=int,
        default=4,
        help="Number of statistical moments (default: 4)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for processing (default: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (default: auto)",
    )

    args = parser.parse_args()

    extract_scattering(
        dataset_path=args.dataset,
        output_dir=args.output,
        stats_path=args.stats,
        J=args.J,
        num_moments=args.moments,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
