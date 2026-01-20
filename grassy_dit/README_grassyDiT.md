```markdown
# GRASSY-DiT

**Graph Diffusion Transformer conditioned on GRASSY Scattering Moments**

Extends [torch-molecule](https://github.com/torch-molecule/torch-molecule)'s Graph-DiT with cross-attention to scattering tokens for inverse molecular design — generating molecules that match target structural signatures.

---

## Overview

GRASSY-DiT combines two key ideas:

1. **GRASSY scattering moments**: A structural fingerprint computed via graph wavelets (dimension varies by dataset)
2. **Graph-DiT**: A discrete diffusion transformer for molecular graph generation

Given a target scattering signature, GRASSY-DiT generates molecules whose structure matches that signature — optionally preserving a scaffold substructure.

```
Target Scattering [auto-dim] + (Optional Scaffold) → GRASSY-DiT → Molecule (SMILES)
```

---

## Architecture

```
                           scattering [B, scatter_dim]
                                   ↓
                           ScatteringTokenizer
                                   ↓
                     tokens [B, num_atom_types + 11, D]
                                   ↓
timestep t → AdaLN ──→ Self-Attn → Cross-Attn → MLP → ... → X, E
                          ↑
                    graph [B, N, D]
```

### Key Components

| Component | Source | Description |
|-----------|--------|-------------|
| `ScatteringTokenizer` | **New** | Dual tokenization: atom-type + level tokens |
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
├── __init__.py               # Package exports
├── model.py                  # ScatteringDenoiser, ScatteringTokenizer, CrossAttention
├── train.py                  # ScatteringGraphDIT subclass + training CLI
├── sample.py                 # Sampling CLI with scaffold support
├── extract_scattering_fixed.py  # Extract scattering from dataset
└── data/                     # Training data (after extraction)
    ├── molecules.csv             # SMILES strings
    └── scattering_moments.npy    # [N, scatter_dim] scattering vectors
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
pip install rdkit pandas numpy tqdm
```

---

## Data Preparation

### Step 1: Prepare Dataset

For MOSES dataset:
```bash
# Downloads MOSES data and computes properties
python datasets/prepare_moses_joao.py --subset 12000
# Creates: datasets/MOSES_12K.npy, datasets/MOSES_12K_stats.npy
```

### Step 2: Extract Scattering Moments

```bash
python grassy_dit/extract_scattering_fixed.py \
    --dataset datasets/MOSES_12K.npy \
    --stats datasets/MOSES_12K_stats.npy \
    --output grassy_dit/data/ \
    --batch_size 64
```

This will:
1. **Auto-detect atom types** from the dataset (e.g., `['Br', 'C', 'Cl', 'F', 'N', 'O', 'S']` for MOSES)
2. **Compute scattering** with matching dimensions
3. **Save** `molecules.csv` and `scattering_moments.npy`

Example output:
```
Detected 7 atom types: ['Br', 'C', 'Cl', 'F', 'N', 'O', 'S']
Scattering configuration:
  - Wavelet scales (J): 4
  - Moments: 4
  - Output dimension: 308
```

---

## Usage

### Training

```bash
# Quick test (CPU, tiny model)
python -m grassy_dit.train \
    --data_dir grassy_dit/data \
    --epochs 1 \
    --batch_size 16 \
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

The model **auto-detects `num_atom_types`** from the scattering dimension at training time.

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
| `--resume_from_checkpoint` | None | Path to checkpoint to resume from |

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
    --molecule "COc1ccccc1N" \
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
| `--molecule` | None | Full molecule SMILES (required for scaffold/remove modes) |
| `--scaffold` | None | SMARTS pattern to preserve during generation |
| `--remove-atoms` | None | Comma-separated atom indices to remove |
| `--output` | generated.txt | Output file for SMILES |

---

## How It Works

### Scattering Moments (Dynamic Dimension)

The scattering dimension is computed as:

```
scatter_dim = num_atom_types × num_levels × num_moments
            = num_atom_types × 11 × 4

Examples:
- MOSES (7 types: Br, C, Cl, F, N, O, S):  7 × 11 × 4 = 308
- ZINC  (9 types):                          9 × 11 × 4 = 396
- Custom dataset: auto-detected from SMILES
```

**Atom types**: Auto-detected by scanning all molecules in the dataset  
**Levels (11)**: 1 zeroth + 4 first + 6 second order wavelets  
**Moments (4)**: mean, variance, skew, kurtosis

### Dual Tokenization

The scattering vector is reshaped into two overlapping views:

```python
# Atom tokens: "What's each atom type doing across all scales?"
atom_tokens = scattering.view(B, num_atom_types, 44)  → Linear → [B, A, D]

# Level tokens: "What's happening at each scale?"  
level_tokens = scattering.view(B, 11, num_atom_types*4) → Linear → [B, 11, D]

# Concat → [B, A + 11, D]
```

### Cross-Attention

Graph tokens (atoms being generated) attend to scattering tokens:

```python
Q = graph_tokens      # [B, N, D] - "What should I be?"
K, V = scatter_tokens # [B, A+11, D] - "Here's the target structure"
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
1. Convert full molecule to graph, identify scaffold atoms via SMARTS matching
2. At each reverse diffusion step, after the model predicts denoised atoms/bonds:
   - Overwrite scaffold positions with clean scaffold values
   - Let non-scaffold positions denoise normally
3. Result: Generated atoms "grow around" the fixed scaffold

Example: Generate molecules containing a benzene ring that match target scattering:
```bash
python -m grassy_dit.sample \
    --checkpoint model.pt \
    --scattering target.npy \
    --molecule "COc1ccccc1N" \
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

1. **molecules.csv**: CSV with a `smiles` column
2. **scattering_moments.npy**: `[N, scatter_dim]` array of scattering moments

⚠️ **Important**: 
- Scattering data should not contain NaN values
- SMILES must be RDKit-parseable
- Scattering dimension must equal `num_atom_types × 11 × 4`

```python
# Check for NaNs
import numpy as np
x = np.load('scattering_moments.npy')
print(f'Shape: {x.shape}')  # e.g., (12000, 308)
print(f'NaNs: {np.isnan(x).sum()}')  # Should be 0
```

---

## Full Pipeline Example

```bash
# 1. Prepare MOSES dataset (12K subset)
python datasets/prepare_moses_joao.py --subset 12000

# 2. Extract scattering moments
python grassy_dit/extract_scattering_fixed.py \
    --dataset datasets/MOSES_12K.npy \
    --stats datasets/MOSES_12K_stats.npy \
    --output grassy_dit/data/

# 3. Train model
python -m grassy_dit.train \
    --data_dir grassy_dit/data \
    --epochs 100 \
    --checkpoint grassy_dit_checkpoint.pt

# 4. Generate molecules
python -m grassy_dit.sample \
    --checkpoint grassy_dit_checkpoint.pt \
    --scattering grassy_dit/data/scattering_moments.npy \
    --index 0 \
    --num_samples 10
```

---

## Known Limitations

1. **Bond types**: Only SINGLE, DOUBLE, TRIPLE, AROMATIC (no DATIVE)
2. **Atom types**: Limited to types present in training data
3. **Molecule size**: Max ~50 atoms (configurable via `--max_node`)
4. **Training data**: Model quality depends heavily on scattering data quality
5. **Scaffold**: Must be valid SMARTS pattern parseable by RDKit

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
```