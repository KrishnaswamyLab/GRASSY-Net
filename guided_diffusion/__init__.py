"""
Scattering Moment-Guided Diffusion for Molecular Generation.

This module provides gradient-based guidance for steering graph diffusion models
toward generating molecules with specific scattering moments, with optional
scaffold constraints to keep core structures fixed.
"""

from .guidance import ScatteringMomentGuidance, apply_guidance_to_probs
from .generator import GuidedGraphDIT, smiles_to_scaffold
from .soft_scattering import DenseSoftScattering, GumbelSoftmaxSampler

__all__ = [
    "ScatteringMomentGuidance",
    "apply_guidance_to_probs",
    "GuidedGraphDIT", 
    "smiles_to_scaffold",
    "DenseSoftScattering",
    "GumbelSoftmaxSampler",
]
