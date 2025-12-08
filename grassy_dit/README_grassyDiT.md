# GRASSY-DiT

**Graph Diffusion Transformer conditioned on GRASSY Scattering Moments**

Extends [torch-molecule](https://github.com/torch-molecule/torch-molecule)'s Graph-DiT with cross-attention to scattering tokens for inverse molecular scattering — generating molecules that match target structural signatures.

---

## Overview

GRASSY-DiT combines two key ideas:
1. **GRASSY scattering moments**: A 440-dimensional structural fingerprint computed via learnable graph wavelets
2. **Graph-DiT**: A discrete diffusion transformer for molecular graph generation

Given a target scattering signature, GRASSY-DiT generates molecules whose structure matches that signature.

```
Target Scattering [440-D] → GRASSY-DiT → Molecule (atoms + bonds)
```

---

## Architecture

```
                           scattering [B, 440]
                                   ↓
                           ScatteringTokenizer
                                   ↓
                            tokens [B, 21, D]
                                   ↓
timestep t → AdaLN ──→ Self-Attn → Cross-Attn → MLP → ... → X, E
                          ↑
                    graph [B, N, D]
```

### Key Components

| Component | Source | Description |
|-----------|--------|-------------|
| `ScatteringTokenizer` | **New** | Dual tokenization: 10 atom-type + 11 level tokens |
| `CrossAttention` | **New** | Graph tokens attend to scattering tokens |
| `SELayerWithCrossAttention` | **New** | DiT block with cross-attention added |
| `ScatteringDenoiser` | **New** | Main model combining all components |
| `Attention` | torch-molecule | Multi-head self-attention with QK norm |
| `MLP` | torch-molecule | Feed-forward network |
| `TimestepEmbedder` | torch-molecule | Sinusoidal timestep encoding |
| `FinalLayer` | torch-molecule | Output projection to atom/bond logits |
| `NoiseScheduleDiscrete` | torch-molecule | Cosine noise schedule for discrete diffusion |
| `MarginalTransition` | torch-molecule | Transition matrices for categorical noise |

---

## What We Inherit from torch-molecule

### Training Infrastructure
- **Discrete diffusion**: Proper categorical noise via transition matrices (not continuous Gaussian)
- **Cosine noise schedule**: Better than linear for discrete data
- **Training loop**: `GraphDITMolecularGenerator.fit()` handles batching, optimization, logging
- **Data processing**: SMILES → graph conversion, padding, masking
- **Checkpointing**: `.save_to_local()` / `.load_from_local()`

### Model Components
```python
from torch_molecule.generator.graph_dit.transformer import (
    AttentionWithNodeMask,  # Self-attention with node masking
    MLP,                    # Feed-forward network
    TimestepEmbedder,       # t → sinusoidal → MLP → [B, D]
    FinalLayer,             # Hidden → atom logits + bond logits
)
```

### Why Subclass Instead of Rewrite?
| Aspect | Custom Implementation | Subclass torch-molecule |
|--------|----------------------|------------------------|
| Lines of code | ~300+ | ~100 |
| Noise schedule | Must implement | Inherited (cosine) |
| Discrete diffusion | Must implement | Inherited (transition matrices) |
| Loss computation | Must implement | Inherited (masked cross-entropy) |
| Bugs | Must debug | Already tested |

---

## What We Extend/Customize

### 1. ScatteringTokenizer — Dual Tokenization

The 440-D scattering vector has structure: `[10 atom types] × [11 levels] × [4 moments]`

We create **two overlapping views**:

```python
# Atom tokens: "What's each atom type doing across all scales?"
atom_tokens = scattering.view(B, 10, 44)  # [B, 10, 11*4]
atom_tokens = Linear(44 → D)              # [B, 10, D]

# Level tokens: "What's happening at each scale across all atoms?"  
level_tokens = scattering.view(B, 11, 40)  # [B, 11, 10*4]
level_tokens = Linear(40 → D)              # [B, 11, D]

# Concat → [B, 21, D]
tokens = cat([atom_tokens, level_tokens])
```

**Why overlap?** The model can query by atom type ("show me carbon's signature") OR by scale ("show me second-order correlations"). Same data, two access patterns.

### 2. CrossAttention — Graph ← Scattering

```python
class CrossAttention(nn.Module):
    """Graph tokens (Q) attend to scattering tokens (K, V)."""
    
    def forward(self, x, context):
        # x: [B, N, D] graph tokens (atoms)
        # context: [B, 21, D] scattering tokens
        
        Q = self.q(x)        # from graph
        K = self.k(context)  # from scattering  
        V = self.v(context)  # from scattering
        
        return scaled_dot_product_attention(Q, K, V)
```

Each atom can ask: "What should I be, given this target scattering?"

### 3. SELayerWithCrossAttention — Modified DiT Block

Original DiT block:
```
Self-Attn → MLP (both modulated by AdaLN)
```

Our block:
```
Self-Attn → Cross-Attn → MLP (self-attn and MLP modulated by AdaLN)
```

### 4. ScatteringGraphDIT — Subclass with Custom Model

```python
class ScatteringGraphDIT(GraphDITMolecularGenerator):
    
    def _validate_inputs(self, X, y, ...):
        # Bypass validation that rejects 440-D conditioning
        return X, y
    
    def _initialize_model(self, model_class, checkpoint=None):
        # Replace their Transformer with our ScatteringDenoiser
        denoiser = ScatteringDenoiser(
            max_n_nodes=self.max_node,
            hidden_size=self.hidden_size,
            depth=self.num_layer,
            num_heads=self.num_head,
            Xdim=self.input_dim_X,  # computed from data
            Edim=self.input_dim_E,  # computed from data
        )
        self.model = ScatteringTransformerAdapter(denoiser)
        return self.model
```

### 5. ScatteringTransformerAdapter — Interface Bridge

torch-molecule's training loop calls:
```python
model.forward(noisy_data, unconditioned)
model.compute_loss(noisy_data, true_X, true_E, ...)
```

Our `ScatteringDenoiser` expects:
```python
model(X_t, E_t, node_mask, t, scattering, uncond)
```

The adapter translates:
```python
class ScatteringTransformerAdapter(nn.Module):
    def forward(self, noisy_data, unconditioned):
        X_t = noisy_data['X_t']
        E_t = noisy_data['E_t']
        scattering = noisy_data['y_t']  # 440-D passed as 'y'
        ...
        return self.denoiser(X_t, E_t, node_mask, t, scattering, uncond=unconditioned)
```

---

## File Structure

```
grassy_dit/
├── __init__.py           # Package exports
├── model.py              # ScatteringDenoiser, ScatteringTokenizer, CrossAttention
├── train.py              # ScatteringGraphDIT subclass + training script
├── scaffold.py           # ScaffoldSampler for constrained generation
├── extract_scattering.py # Extract moments from trained GRASSY model
└── data/                 # Training data (created by extract_scattering.py)
    ├── molecules.csv         # SMILES strings
    └── scattering_moments.npy # [N, 440] scattering vectors
```

---

## Installation

### Prerequisites
```bash
# Python 3.9-3.10 (required for torch-molecule compatibility)
conda create -n grassy_dit python=3.10
conda activate grassy_dit

# PyTorch 2.0.0 with CUDA 11.8 (must match torch-molecule requirements)
pip install torch==2.0.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install torchvision==0.15.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric

# torch-molecule
pip install torch-molecule

# Other dependencies
pip install rdkit pysmiles pandas numpy
```

### Verify Installation
```python
import torch
from torch_molecule import GraphDITMolecularGenerator
from torch_molecule.generator.graph_dit.transformer import AttentionWithNodeMask
print("✓ torch-molecule installed correctly")
```

---

## Usage

### 1. Extract Scattering Moments

First, compute scattering moments for your molecules using a trained GRASSY model:

```bash
mkdir -p grassy_dit/data
python -m grassy_dit.extract_scattering
```

This creates:
- `grassy_dit/data/scattering_moments.npy` — [N, 440] array
- `grassy_dit/data/molecules.csv` — SMILES strings

### 2. Training

```bash
# Quick sanity check (CPU, small model)
python -m grassy_dit.train \
    --data_dir grassy_dit/data \
    --epochs 1 \
    --batch_size 4 \
    --num_layer 2 \
    --hidden_size 64

# Full training (GPU recommended)
python -m grassy_dit.train \
    --data_dir grassy_dit/data \
    --epochs 100 \
    --batch_size 32 \
    --num_layer 12 \
    --hidden_size 384 \
    --num_head 16
```

### 3. Sampling

```python
from grassy_dit.train import ScatteringGraphDIT
from grassy_dit.scaffold import ScaffoldSampler
import numpy as np
import torch

# Load trained model
generator = ScatteringGraphDIT()
generator.load_from_local('grassy_dit_checkpoint.pt')

# Create sampler
sampler = ScaffoldSampler(generator.model.denoiser)

# Target scattering moments
target_scatter = torch.tensor(np.load('grassy_dit/data/scattering_moments.npy')[0])

# Generate unconditionally
X, E = sampler.sample(
    scattering=target_scatter,
    num_atoms=20,
    guidance_scale=2.0,
)

# Generate with scaffold constraint (e.g., keep benzene ring fixed)
X, E = sampler.sample(
    scattering=target_scatter,
    num_atoms=20,
    guidance_scale=2.0,
    scaffold_atoms=[0, 1, 2, 3, 4, 5],
    scaffold_X=benzene_X,
    scaffold_E=benzene_E,
)
```

---

## Scattering Moment Structure

The 440-dimensional scattering vector encodes multi-scale structural information:

```
440 = 10 atom types × 11 levels × 4 moments

Atom types (10):
  C, N, O, F, S, Cl, Br, ... (dataset-dependent)

Levels (11):
  - 1 zeroth order: raw node features
  - 4 first order: wavelet coefficients at scales j=1,2,3,4
  - 6 second order: |Ψ_j' |Ψ_j x|| at scale pairs (feng_filters)

Moments (4):
  - mean: average over nodes
  - variance: spread
  - skew: asymmetry  
  - kurtosis: tail weight
```

The moments aggregate per-node wavelet coefficients into a fixed-size graph-level descriptor, regardless of molecule size.

---

## Classifier-Free Guidance (CFG)

### Training
During training, 10% of batches replace scattering tokens with learned `null` embeddings:
```python
if train and dropout > 0:
    mask = torch.rand(B) < 0.1
    tokens[mask] = self.null.expand(...)
```

This teaches the model to generate both conditionally and unconditionally.

### Sampling
At inference, combine both predictions:
```python
pred_uncond = model(x, scattering, uncond=True)
pred_cond = model(x, scattering, uncond=False)
pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
```

Higher `guidance_scale` (1.5-3.0) produces molecules that more strongly match the target scattering.

---

## Scaffold-Constrained Generation

The `ScaffoldSampler` supports inpainting-style generation where some atoms/bonds are fixed:

```python
# Scaffold atoms stay clean (no noise) throughout denoising
# Generated atoms attend to scaffold via self-attention
# Result: new atoms are compatible with scaffold structure

X, E = sampler.sample(
    scattering=target,
    num_atoms=25,
    scaffold_atoms=[0, 1, 2, 3, 4, 5],  # indices of fixed atoms
    scaffold_X=scaffold_atom_features,   # [max_nodes, Xdim]
    scaffold_E=scaffold_bond_features,   # [max_nodes, max_nodes, Edim]
)
```

---

## Training Tips

### Hyperparameters
| Parameter | Default | Notes |
|-----------|---------|-------|
| `hidden_size` | 384 | Larger = more capacity, slower |
| `num_layer` | 12 | Depth of transformer |
| `num_head` | 16 | Must divide hidden_size |
| `batch_size` | 32 | Larger if GPU memory allows |
| `learning_rate` | 1e-4 | Standard for transformers |
| `epochs` | 100 | Monitor validation loss |

### Data Requirements
- Minimum ~1000 molecules for basic training
- More data = better generalization
- Filter out molecules with unusual bond types (DATIVE bonds not supported)

### Hardware
- **CPU**: Feasible for small models (2-4 layers), very slow for full model
- **GPU**: Recommended for full training (12 layers, 384 hidden)
- **Memory**: ~8GB GPU memory for batch_size=32, full model

---

## Comparison: Property Conditioning vs Scattering Conditioning

| Aspect | Graph-DiT (Original) | GRASSY-DiT (Ours) |
|--------|---------------------|-------------------|
| Conditioning | 1-10 properties (QED, logP, etc.) | 440-D scattering signature |
| Mechanism | Add to timestep embedding | Cross-attention to tokens |
| Information | Scalar targets | Full structural fingerprint |
| Use case | "Generate molecule with QED=0.8" | "Generate molecule with this structure" |

---

## Known Limitations

1. **Bond type support**: Only handles SINGLE, DOUBLE, TRIPLE, AROMATIC (no DATIVE)
2. **Atom types**: Limited to common drug-like atoms (~10 types)
3. **Molecule size**: Max ~50 atoms (configurable but affects memory)
4. **Scattering mismatch**: Model trained on specific dataset may not generalize to very different chemistry

---

## Citation

If you use this code, please cite:

```bibtex
@article{bhaskar2021molecular,
  title={Molecular Graph Generation via Geometric Scattering},
  author={Bhaskar, Dhananjay and Grady, Jackson D and Perlmutter, Michael A and Krishnaswamy, Smita},
  journal={arXiv preprint arXiv:2110.06241},
  year={2021}
}
```

---

## License

MIT License (same as GRASSY-Net)
