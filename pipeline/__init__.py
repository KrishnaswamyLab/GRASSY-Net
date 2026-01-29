"""
GRASSY-Net End-to-End Pipeline

A unified pipeline that orchestrates the entire workflow from raw data to evaluation reports:
1. Data Preparation - Convert SMILES to NPY with properties
2. Scattering Extraction - Compute scattering moments
3. Data Splitting - Split into train/val/test (deterministic with seed)
4. GRASSY Training - Train the autoencoder
5. DiT Training - Train the diffusion transformer
6. Evaluation - Generate molecules and compute metrics

Usage:
    python -m pipeline.run_pipeline --input data.smi
    python -m pipeline.run_pipeline --input data.smi --dit-checkpoint model.pt --dit-epochs 0
"""

from .pipeline_config import PipelineConfig, load_config, save_config
from .checkpoint_manager import CheckpointManager
from .report_generator import ReportGenerator

__version__ = "0.1.0"
__all__ = ["PipelineConfig", "load_config", "save_config", "CheckpointManager", "ReportGenerator"]
