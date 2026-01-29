"""
Pipeline stage wrappers.

Each stage is a self-contained function that:
- Takes input paths and config as arguments
- Returns output paths (checkpoint, data directory, etc.)
- Respects checkpoint injection and epochs=0 skip behavior
"""

from .data_prep import run_data_prep
from .scattering import run_scattering
from .splitting import run_splitting
from .train_grassy import run_train_grassy
from .train_dit import run_train_dit
from .evaluate import run_evaluate
from .sample_unconstrained import run_sample_unconstrained
from .sample_property_opt import run_sample_property_opt

__all__ = [
    "run_data_prep",
    "run_scattering", 
    "run_splitting",
    "run_train_grassy",
    "run_train_dit",
    "run_evaluate",
    "run_sample_unconstrained",
    "run_sample_property_opt",
]
