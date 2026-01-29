"""
Stage 2: Scattering Feature Extraction

Extracts scattering moments from molecular graphs using the fixed GraphScatteringTransform.
Wraps grassy_dit/extract_scattering_fixed.py functionality.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.ScatteringTransform import GraphScatteringTransform
from datasets.ZINCDataset import ZINCDataset


def run_scattering(
    data_path: str,
    stats_path: str,
    output_dir: str,
    J: int = 4,
    num_moments: int = 4,
    batch_size: int = 64,
    device: str = "auto",
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Run scattering extraction stage.
    
    Args:
        data_path: Path to dataset .npy file
        stats_path: Path to statistics .npy file
        output_dir: Directory to save outputs
        J: Number of wavelet scales
        num_moments: Number of statistical moments
        batch_size: Batch size for processing
        device: Device to use ("auto", "cuda", "cpu")
    
    Returns:
        Tuple of (scattering_path, molecules_csv_path, metrics_dict)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Stage 2: Scattering Feature Extraction")
    print("=" * 60)
    
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
    print(f"\nLoading dataset from: {data_path}")
    dataset = ZINCDataset(data_path, prop_stat_dict=stats_path)
    print(f"Loaded {len(dataset)} molecules")
    print(f"Node features: {dataset.num_node_features}")
    
    # Create scattering transform
    scattering = GraphScatteringTransform(
        in_channels=dataset.num_node_features,
        J=J,
        num_moments=num_moments,
    ).to(device)
    scattering.eval()
    
    scattering_dim = scattering.out_shape()
    print(f"\nScattering configuration:")
    print(f"  - Wavelet scales (J): {J}")
    print(f"  - Moments: {num_moments}")
    print(f"  - Output dimension: {scattering_dim}")
    
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
    print(f"Scattering coefficients shape: {scattering_coeffs.shape}")
    
    # Get SMILES list
    all_smiles = dataset.smi
    print(f"SMILES count: {len(all_smiles)}")
    
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
    
    # Metrics for reporting
    metrics = {
        'num_molecules': len(all_smiles),
        'scattering_dim': scattering_dim,
        'J': J,
        'num_moments': num_moments,
        'nan_count': int(nan_count),
    }
    
    print(f"\nScattering extraction complete!")
    return scattering_path, molecules_path, metrics
