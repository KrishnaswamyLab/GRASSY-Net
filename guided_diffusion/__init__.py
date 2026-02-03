"""
Scattering Moment-Guided Diffusion for Molecular Generation.

This module provides gradient-based guidance for steering graph diffusion models
toward generating molecules with specific scattering moments, with optional
scaffold constraints to keep core structures fixed.
"""

from .guidance import ScatteringMomentGuidance
from .generator import GuidedGraphDIT, smiles_to_scaffold

__all__ = ["ScatteringMomentGuidance", "GuidedGraphDIT", "smiles_to_scaffold"]
