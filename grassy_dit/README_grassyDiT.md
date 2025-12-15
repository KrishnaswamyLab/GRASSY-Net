# GRASSY-DiT

**Graph Diffusion Transformer conditioned on GRASSY Scattering Moments**

Extends [torch-molecule](https://github.com/torch-molecule/torch-molecule)'s Graph-DiT with cross-attention to scattering tokens for inverse molecular design — generating molecules that match target structural signatures.

---

## Overview

GRASSY-DiT combines two key ideas:

1. **GRASSY scattering moments**: A 440-dimensional structural fingerprint computed via learnable graph wavelets
2. **Graph-DiT**: A discrete diffusion transformer for molecular graph generation

Given a target scattering signature, GRASSY-DiT generates molecules whose structure matches that signature — optionally preserving a scaffold substructure.

```
Target Scattering [440-D] + (Optional Scaffold) → GRASSY-DiT → Molecule (SMILES)
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
| `NoiseScheduleDiscrete` | torch-molecule | Cosine noise schedule for discrete diffusion |
| `MarginalTransition` | torch-molecule | Transition matrices for categorical noise |
| `FinalLayer` | torch-molecule | Output projection to atom/bond logits |

---

## File Structure

```
grassy_dit/
├── __init__.py           # Package exports
├── model.py              # ScatteringDenoiser, ScatteringTokenizer, CrossAttention
├── train.py              # ScatteringGraphDIT subclass + training CLI
├── sample.py             # Sampling CLI with scaffold support
└── data/                 # Training data
    ├── molecules.csv         # SMILES strings
    └── scattering_moments.npy # [N, 440] scattering vectors
```

---

## Installation

```bash
# Python 3.9-3.10
conda create -n grassy_dit python=3.10
conda activate grassy_dit

# PyTorch 2.0.0
pip install torch==2.0.0

# PyTorch Geometric
pip install torch-scatter torch-sparse torch-geometric

# torch-molecule
pip install torch-molecule

# Other dependencies
pip install rdkit pandas numpy
```

---

## Usage

### Training

```bash
# Quick test (CPU, tiny model)
python -m grassy_dit.train \
    --data_dir grassy_dit/data \
    --epochs 1 \
    --batch_size 2 \
    --hidden_size 64 \
    --num_layer 2 \
    --num_head 4 \
    --checkpoint test_checkpoint.pt

# Full training (GPU recommended)
python -m grassy_dit.train \
    --data_dir grassy_dit/data \
    --epochs 100 \
    --batch_size 32 \
    --hidden_size 384 \
    --num_layer 12 \
    --num_head 16 \
    --checkpoint grassy_dit_checkpoint.pt
```

**Training arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | required | Directory containing molecules.csv and scattering_moments.npy |
| `--csv_file` | molecules.csv | CSV filename with SMILES |
| `--smiles_col` | smiles | Column name for SMILES |
| `--scatter_file` | scattering_moments.npy | Scattering moments filename |
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--hidden_size` | 384 | Transformer hidden dimension |
| `--num_layer` | 12 | Number of transformer layers |
| `--num_head` | 16 | Number of attention heads |
| `--lr` | 1e-4 | Learning rate |
| `--checkpoint` | grassy_dit_checkpoint.pt | Output checkpoint path |

### Sampling

```bash
# Generate 10 molecules conditioned on a scattering vector
python -m grassy_dit.sample \
    --checkpoint grassy_dit_checkpoint.pt \
    --scattering target_scattering.npy \
    --num_samples 10 \
    --output generated.txt

# Use specific scattering from a multi-row file
python -m grassy_dit.sample \
    --checkpoint grassy_dit_checkpoint.pt \
    --scattering grassy_dit/data/scattering_moments.npy \
    --index 42 \
    --num_samples 5 \
    --output generated.txt

# Generate with scaffold constraint (e.g., keep benzene ring)
python -m grassy_dit.sample \
    --checkpoint grassy_dit_checkpoint.pt \
    --scattering target_scattering.npy \
    --scaffold "c1ccccc1" \
    --num_samples 5 \
    --output generated.txt
```

**Sampling arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | required | Path to trained model |
| `--scattering` | required | Path to .npy file with target scattering |
| `--index` | 0 | Index of scattering vector (if file has multiple rows) |
| `--num_samples` | 10 | Number of molecules to generate |
| `--num_nodes` | None | Number of atoms (None = sample from training distribution) |
| `--scaffold` | None | Scaffold SMILES to preserve during generation |
| `--output` | generated.txt | Output file for SMILES |

---

## How It Works

### Scattering Moments (440-D)

```
440 = 10 atom types × 11 levels × 4 moments

Atom types (10): C, N, O, F, S, Cl, Br, ...
Levels (11): 1 zeroth + 4 first + 6 second order wavelets
Moments (4): mean, variance, skew, kurtosis
```

### Dual Tokenization

The 440-D vector is reshaped into two overlapping views:

```python
# Atom tokens: "What's each atom type doing across all scales?"
atom_tokens = scattering.view(B, 10, 44)  → Linear → [B, 10, D]

# Level tokens: "What's happening at each scale?"  
level_tokens = scattering.view(B, 11, 40) → Linear → [B, 11, D]

# Concat → [B, 21, D]
```

### Cross-Attention

Graph tokens (atoms being generated) attend to scattering tokens:

```python
Q = graph_tokens      # [B, N, D] - "What should I be?"
K, V = scatter_tokens # [B, 21, D] - "Here's the target structure"
output = softmax(Q @ K.T) @ V
```

### Classifier-Free Guidance (CFG)

During training, 10% of batches use null conditioning. At inference:

```python
pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
```

Higher `guidance_scale` (default 2.0) = stronger adherence to target scattering.

### Scaffold-Constrained Generation

Scaffold constraints allow preserving a substructure during generation:

```
Position:        0  1  2  3  4  5  6  7  8  9  10 ...
scaffold_mask:   T  T  T  T  T  T  F  F  F  F  F  ...
node_mask:       T  T  T  T  T  T  T  T  T  T  T  ...
                 |--scaffold---|  |--generated--|
```

How it works:
1. Convert scaffold SMILES to atom/bond tensors using model's atom/bond decoders
2. At each reverse diffusion step, after the model predicts denoised atoms/bonds:
   - Overwrite scaffold positions with clean scaffold values
   - Let non-scaffold positions denoise normally
3. Result: Generated atoms "grow around" the fixed scaffold

Example: Generate molecules containing a benzene ring that match target scattering:
```bash
python -m grassy_dit.sample \
    --checkpoint model.pt \
    --scattering target.npy \
    --scaffold "c1ccccc1" \
    --num_samples 10
```

---

## What We Inherit from torch-molecule

- **Discrete diffusion**: Categorical noise via transition matrices
- **Training loop**: Batching, optimization, checkpointing
- **Data processing**: SMILES → graph conversion
- **Sampling**: Reverse diffusion with proper posterior computation
- **SMILES conversion**: Tensor → valid molecules

---

## Data Requirements

Your `data_dir` should contain:

1. **molecules.csv**: CSV with a SMILES column
2. **scattering_moments.npy**: `[N, 440]` array of scattering moments

⚠️ **Important**: Scattering data should not contain NaN values. Filter them before training.

```python
# Check for NaNs
import numpy as np
x = np.load('scattering_moments.npy')
print(f'NaNs: {np.isnan(x).sum()}')  # Should be 0
```

---

## Known Limitations

1. **Bond types**: Only SINGLE, DOUBLE, TRIPLE, AROMATIC (no DATIVE)
2. **Atom types**: ~10 common drug-like atoms
3. **Molecule size**: Max ~50 atoms (configurable)
4. **Training data**: Model quality depends heavily on scattering data quality
5. **Scaffold**: Must be valid SMILES parseable by RDKit

---

## Citation

```bibtex
@article{bhaskar2021molecular,
  title={Molecular Graph Generation via Geometric Scattering},
  author={Bhaskar, Dhananjay and Grady, Jackson D and Perlmutter, Michael A and Krishnaswamy, Smita},
  journal={arXiv preprint arXiv:2110.06241},
  year={2021}
}
```