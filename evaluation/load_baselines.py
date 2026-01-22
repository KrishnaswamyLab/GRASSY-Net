"""
Load baseline samples and metrics from MOSES and DiGress.

Provides utilities to load pre-generated samples from MOSES baselines
(VAE, AAE, CharRNN, JTN, LatentGAN, NGram, HMM, Combinatorial)
and generate samples from DiGress for comparison.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import subprocess

# Paths
EVALUATION_DIR = Path(__file__).parent
PROJECT_ROOT = EVALUATION_DIR.parent
MOSES_DIR = PROJECT_ROOT / "external" / "moses"
MOSES_DATA_DIR = MOSES_DIR / "data"
MOSES_SAMPLES_DIR = MOSES_DATA_DIR / "samples"
DIGRESS_CHECKPOINT_DIR = EVALUATION_DIR / "checkpoints" / "digress"

# Available MOSES baselines
MOSES_BASELINES = [
    "vae",
    "aae",
    "char_rnn",
    "jtn",
    "latent_gan",
    "ngram",
    "hmm",
    "combinatorial",
]

#### loading if needed 
# MOSES dataset data directory (inside moses package)
MOSES_DATASET_DIR = MOSES_DIR / "moses" / "dataset" / "data"

MOSES_DATA_URLS = {
    "train.csv.gz": "https://media.githubusercontent.com/media/molecularsets/moses/master/moses/dataset/data/train.csv.gz",
    "test.csv.gz": "https://media.githubusercontent.com/media/molecularsets/moses/master/moses/dataset/data/test.csv.gz",
    "test_scaffolds.csv.gz": "https://media.githubusercontent.com/media/molecularsets/moses/master/moses/dataset/data/test_scaffolds.csv.gz",
    "test_stats.npz": "https://media.githubusercontent.com/media/molecularsets/moses/master/moses/dataset/data/test_stats.npz",
    "test_scaffolds_stats.npz": "https://media.githubusercontent.com/media/molecularsets/moses/master/moses/dataset/data/test_scaffolds_stats.npz",
}


def ensure_moses_data(filename: str):
    """Auto-download MOSES data file if missing or is Git LFS pointer."""
    filepath = MOSES_DATASET_DIR / filename
    
    if filepath.exists():
        with open(filepath, 'rb') as f:
            header = f.read(7)
        if header == b'version':
            print(f"Detected Git LFS pointer for {filename}, downloading...")
            filepath.unlink()
    
    if not filepath.exists():
        if filename not in MOSES_DATA_URLS:
            raise FileNotFoundError(f"No download URL for {filename}")
        print(f"Downloading {filename}...")
        subprocess.run(["curl", "-L", "-o", str(filepath), MOSES_DATA_URLS[filename]], check=True)
        print(f"Downloaded to {filepath}")


def ensure_moses_stats():
    """Auto-download MOSES statistics and test data files if missing."""
    ensure_moses_data("test_stats.npz")
    ensure_moses_data("test_scaffolds_stats.npz")
    ensure_moses_data("test.csv.gz")
    ensure_moses_data("test_scaffolds.csv.gz")


# Baseline metrics URLs (GitHub media for Git LFS files)
MOSES_BASELINE_URL_TEMPLATE = "https://media.githubusercontent.com/media/molecularsets/moses/master/data/samples/{model}/metrics_{model}_{run_id}.csv"


def ensure_baseline_metrics(model_name: str, run_id: int = 1):
    """Auto-download baseline metrics file if missing or is Git LFS pointer."""
    filename = f"metrics_{model_name}_{run_id}.csv"
    filepath = MOSES_SAMPLES_DIR / model_name / filename
    
    # Create directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists and is not a Git LFS pointer
    if filepath.exists():
        with open(filepath, 'rb') as f:
            header = f.read(7)
        if header == b'version':
            print(f"Detected Git LFS pointer for {filename}, downloading...")
            filepath.unlink()
    
    # Download if missing
    if not filepath.exists():
        url = MOSES_BASELINE_URL_TEMPLATE.format(model=model_name, run_id=run_id)
        print(f"Downloading {filename}...")
        subprocess.run(["curl", "-L", "-o", str(filepath), url], check=True)
        print(f"Downloaded to {filepath}")


#####

def load_baseline_samples(
    model_name: str,
    run_id: int = 1,
    max_samples: Optional[int] = None,
) -> List[str]:
    """
    Load pre-generated samples from a MOSES baseline model.

    Args:
        model_name: Name of the baseline model (e.g., 'vae', 'aae', 'char_rnn')
        run_id: Run number (1, 2, or 3 for individual runs, or 'all' for combined)
        max_samples: Maximum number of samples to return (None = all)

    Returns:
        List of SMILES strings

    Example:
        >>> samples = load_baseline_samples('vae', run_id=1)
        >>> print(len(samples))
        30000
    """
    model_name = model_name.lower().replace("-", "_")

    if model_name not in MOSES_BASELINES:
        raise ValueError(
            f"Unknown baseline model: {model_name}. "
            f"Available: {MOSES_BASELINES}"
        )

    # Determine file path
    if run_id == "all":
        filename = f"{model_name}_all.csv"
    else:
        filename = f"{model_name}_{run_id}.csv"

    filepath = MOSES_SAMPLES_DIR / model_name / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Baseline samples not found at: {filepath}\n"
            f"Make sure the MOSES samples are downloaded. "
            f"Check external/moses/data/samples/"
        )

    # Load samples
    df = pd.read_csv(filepath)

    # Handle different column names
    if "SMILES" in df.columns:
        smiles = df["SMILES"].tolist()
    elif "smiles" in df.columns:
        smiles = df["smiles"].tolist()
    elif df.shape[1] == 1:
        # Single column, assume it's SMILES
        smiles = df.iloc[:, 0].tolist()
    else:
        raise ValueError(f"Cannot determine SMILES column in {filepath}")

    if max_samples is not None:
        smiles = smiles[:max_samples]

    return smiles


def load_baseline_metrics(
    model_name: str,
    run_id: int = 1,
) -> Dict[str, float]:
    """
    Load pre-computed metrics for a MOSES baseline model.

    Args:
        model_name: Name of the baseline model
        run_id: Run number (1, 2, or 3)

    Returns:
        Dictionary of metric names to values

    Example:
        >>> metrics = load_baseline_metrics('vae', run_id=1)
        >>> print(metrics['valid'])
        0.977
    """
    model_name = model_name.lower().replace("-", "_")

    if model_name not in MOSES_BASELINES:
        raise ValueError(
            f"Unknown baseline model: {model_name}. "
            f"Available: {MOSES_BASELINES}"
        )

    filename = f"metrics_{model_name}_{run_id}.csv"
    filepath = MOSES_SAMPLES_DIR / model_name / filename

    # Auto-download if missing or Git LFS pointer
    ensure_baseline_metrics(model_name, run_id)

    df = pd.read_csv(filepath)

    # Convert to dictionary (handle both row and column formats)
    if df.shape[0] == 1:
        # Metrics as columns
        return df.iloc[0].to_dict()
    else:
        # Metrics as rows with name/value columns
        if "metric" in df.columns and "value" in df.columns:
            return dict(zip(df["metric"], df["value"]))
        else:
            # Assume first column is metric name, second is value
            return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))


def load_all_baseline_metrics(run_id: int = 1) -> Dict[str, Dict[str, float]]:
    """
    Load metrics for all available MOSES baselines.

    Args:
        run_id: Run number (1, 2, or 3)

    Returns:
        Dictionary mapping model names to their metrics

    Example:
        >>> all_metrics = load_all_baseline_metrics()
        >>> print(all_metrics['vae']['valid'])
        0.977
    """
    results = {}

    for model_name in MOSES_BASELINES:
        try:
            results[model_name] = load_baseline_metrics(model_name, run_id)
        except FileNotFoundError:
            print(f"Warning: Could not load metrics for {model_name}")

    return results


def load_moses_train_set() -> List[str]:
    """
    Load the MOSES training set SMILES.

    Returns:
        List of training SMILES strings
    """
    # Try using the moses dataset module first
    ensure_moses_data("train.csv.gz")
    try:
        sys.path.insert(0, str(MOSES_DIR))
        from moses.dataset import get_dataset

        return list(get_dataset("train"))
    except ImportError:
        pass

    # Fallback: load directly from file
    train_path = MOSES_DATA_DIR / "train.csv"
    if train_path.exists():
        df = pd.read_csv(train_path)
        col = "SMILES" if "SMILES" in df.columns else "smiles"
        return df[col].tolist()

    # Try gzipped version
    train_gz_path = MOSES_DATA_DIR / "train.csv.gz"
    if train_gz_path.exists():
        df = pd.read_csv(train_gz_path, compression="gzip")
        col = "SMILES" if "SMILES" in df.columns else "smiles"
        return df[col].tolist()

    raise FileNotFoundError(
        "MOSES training set not found. Check external/moses/data/"
    )


def load_moses_test_set(scaffold: bool = False) -> List[str]:
    """
    Load the MOSES test set SMILES.

    Args:
        scaffold: If True, load scaffold test set instead of standard test

    Returns:
        List of test SMILES strings
    """
    filename_gz = "test_scaffolds.csv.gz" if scaffold else "test.csv.gz"
    ensure_moses_data(filename_gz)
    try:
        sys.path.insert(0, str(MOSES_DIR))
        from moses.dataset import get_dataset

        split = "test_scaffolds" if scaffold else "test"
        return list(get_dataset(split))
    except ImportError:
        pass

    # Fallback: load directly
    filename = "test_scaffolds.csv" if scaffold else "test.csv"
    test_path = MOSES_DATA_DIR / filename
    if test_path.exists():
        df = pd.read_csv(test_path)
        col = "SMILES" if "SMILES" in df.columns else "smiles"
        return df[col].tolist()

    raise FileNotFoundError(f"MOSES test set not found: {test_path}")


def check_digress_available() -> bool:
    """
    Check if DiGress checkpoints are available.

    Returns:
        True if DiGress checkpoint found, False otherwise
    """
    if not DIGRESS_CHECKPOINT_DIR.exists():
        return False

    # Look for any checkpoint files
    checkpoint_files = list(DIGRESS_CHECKPOINT_DIR.glob("*.pt")) + \
                      list(DIGRESS_CHECKPOINT_DIR.glob("*.ckpt"))

    return len(checkpoint_files) > 0


def load_digress_model(device: str = "cuda"):
    """
    Load DiGress model from checkpoint for comparison.

    Note: This requires DiGress to be installed and checkpoint available.

    Args:
        device: Device to load model on

    Returns:
        DiGress model ready for generation

    Raises:
        FileNotFoundError: If no DiGress checkpoint found
        ImportError: If DiGress is not installed
    """
    if not check_digress_available():
        raise FileNotFoundError(
            f"No DiGress checkpoint found in {DIGRESS_CHECKPOINT_DIR}\n"
            f"Please place a DiGress checkpoint file (.pt or .ckpt) in this directory."
        )

    # Find checkpoint
    checkpoint_files = list(DIGRESS_CHECKPOINT_DIR.glob("*.pt")) + \
                      list(DIGRESS_CHECKPOINT_DIR.glob("*.ckpt"))
    checkpoint_path = checkpoint_files[0]

    try:
        # Attempt to import DiGress
        # Note: User needs to have DiGress installed
        import digress
        from digress.models import DiGress

        model = DiGress.load_from_checkpoint(checkpoint_path)
        model = model.to(device)
        model.eval()
        return model
    except ImportError:
        raise ImportError(
            "DiGress is not installed. To compare with DiGress:\n"
            "1. Install DiGress: pip install digress\n"
            "2. Place checkpoint in evaluation/checkpoints/digress/\n"
            "3. Run benchmark with --compare-digress"
        )


def generate_digress_samples(
    num_samples: int = 10000,
    batch_size: int = 64,
    device: str = "cuda",
) -> List[str]:
    """
    Generate samples from DiGress model for comparison.

    Args:
        num_samples: Number of molecules to generate
        batch_size: Generation batch size
        device: Device for generation

    Returns:
        List of generated SMILES strings
    """
    model = load_digress_model(device)

    all_smiles = []
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, num_samples - len(all_smiles))

        # Generate batch
        # Note: Actual API depends on DiGress implementation
        try:
            smiles_batch = model.generate(batch_size=current_batch_size)
            all_smiles.extend(smiles_batch)
        except Exception as e:
            print(f"DiGress generation error: {e}")
            break

        if len(all_smiles) >= num_samples:
            break

    return all_smiles[:num_samples]


def get_available_baselines() -> Dict[str, bool]:
    """
    Check which baselines have samples/metrics available.

    Returns:
        Dictionary mapping baseline names to availability status
    """
    available = {}

    for model_name in MOSES_BASELINES:
        try:
            filepath = MOSES_SAMPLES_DIR / model_name / f"{model_name}_1.csv"
            available[model_name] = filepath.exists()
        except Exception:
            available[model_name] = False

    # Check DiGress
    available["digress"] = check_digress_available()

    return available


def print_available_baselines():
    """Print a summary of available baselines."""
    available = get_available_baselines()

    print("\n" + "=" * 50)
    print("Available Baselines for Comparison")
    print("=" * 50)

    print("\nMOSES Baselines:")
    for model in MOSES_BASELINES:
        status = "✓ Available" if available.get(model, False) else "✗ Not found"
        print(f"  {model}: {status}")

    print("\nExternal Models:")
    status = "✓ Available" if available.get("digress", False) else "✗ Not found"
    print(f"  DiGress: {status}")

    if not available.get("digress", False):
        print(f"\n  To add DiGress: Place checkpoint in")
        print(f"  {DIGRESS_CHECKPOINT_DIR}/")

    print("=" * 50 + "\n")
