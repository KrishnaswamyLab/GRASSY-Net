"""
Evaluation Benchmarks for GRASSY-DiT.

This module provides tools to evaluate GRASSY-DiT molecular generation:
- MOSES Benchmark: Compare against VAE, AAE, CharRNN, JTN, etc.
- BACE Benchmark: Compare against Graph DiT, DiGress, MOOD

Supports four generation modes:
- unconditional: DiT generates randomly (no conditioning)
- direct: molecule → scattering → DiT
- full-pipeline: molecule → scattering → GRASSY VAE → moments → DiT
- latent-sample: random latent z → GRASSY decode → moments → DiT
"""

from .moses_benchmark import run_benchmark, MOSESBenchmark
from .bace_eval import BACEBenchmark
from .load_baselines import load_baseline_samples, load_baseline_metrics
from .utils import (
    load_dit_model,
    load_grassy_vae,
    load_scattering_model,
    compute_scattering_from_smiles,
    smiles_to_pyg_data,
)

__all__ = [
    "run_benchmark",
    "MOSESBenchmark",
    "BACEBenchmark",
    "load_baseline_samples",
    "load_baseline_metrics",
    "load_dit_model",
    "load_grassy_vae",
    "load_scattering_model",
    "compute_scattering_from_smiles",
    "smiles_to_pyg_data",
]
