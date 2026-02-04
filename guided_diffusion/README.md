# Guided Diffusion for GRASSY-DiT

This module provides **scattering moment guidance** for molecular generation with GRASSY-DiT. It enables steering molecule generation toward target structural properties encoded as scattering moments.

## Overview

Two guidance approaches are implemented:

| Approach | Description | Operates On |
|----------|-------------|-------------|
| **Direct Scattering Guidance** | Computes scattering on predicted clean graph, backprops through differentiable scattering | `pred_X, pred_E` (model predictions) |
| **Classifier Guidance** | Uses trained classifier to predict moments from noisy graphs | `X_t, E_t` (current noisy state) |

Both approaches can be used independently or combined.

## Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │              Reverse Diffusion Loop                      │
                    │                                                          │
                    │   X_T ──► X_{T-1} ──► ... ──► X_1 ──► X_0 (clean)       │
                    │                                                          │
                    │   At each step t:                                        │
                    │   ┌──────────────────────────────────────────────────┐   │
                    │   │  1. Model predicts: pred_X, pred_E              │   │
                    │   │  2. Compute guidance gradients                   │   │
                    │   │  3. Shift predictions toward target moments      │   │
                    │   │  4. Sample next state                            │   │
                    │   └──────────────────────────────────────────────────┘   │
                    └─────────────────────────────────────────────────────────┘
```

## File Structure

```
guided_diffusion/
├── README.md                        # This file
├── __init__.py
│
├── # === DIRECT SCATTERING GUIDANCE (existing) ===
├── guidance.py                      # ScatteringMomentGuidance class
├── generator.py                     # GuidedGraphDIT generator
├── sample.py                        # CLI for direct guidance generation
│
├── # === CLASSIFIER GUIDANCE (new) ===
├── moment_classifier.py             # MomentClassifier architecture
├── prepare_classifier_data.py       # Generate training data for classifier
├── train_classifier.py              # Train the classifier
├── classifier_guidance.py           # ClassifierGuidance class
├── classifier_generator.py          # ClassifierGuidedGraphDIT generator
├── generate_classifier_guided.py    # CLI for classifier guidance generation
│
├── # === EVALUATION ===
├── evaluate_guidance.py             # Compare guidance approaches
├── sweep_guidance_scale.py          # Hyperparameter sweep
├── test_guidance.py                 # Unit tests
│
└── checkpoints/
    └── moment_classifier/           # Trained classifier checkpoints
```

---

## Quick Start

### Direct Scattering Guidance

```bash
# Generate molecules toward target moments (from SMILES)
python -m guided_diffusion.sample \
    --checkpoint path/to/dit_checkpoint.pt \
    --target_smiles "c1ccccc1" \
    --guidance_scale 1.0 \
    --num_samples 100
```

### Classifier Guidance

```bash
# 1. Prepare training data
python -m guided_diffusion.prepare_classifier_data \
    --data_dir grassy_dit/data \
    --output_dir guided_diffusion/classifier_data \
    --num_timesteps_per_mol 10

# 2. Train classifier
python -m guided_diffusion.train_classifier \
    --data_path guided_diffusion/classifier_data/classifier_training_data.pt \
    --output_dir guided_diffusion/checkpoints/moment_classifier \
    --epochs 100

# 3. Generate with classifier guidance
python -m guided_diffusion.generate_classifier_guided \
    --dit_checkpoint path/to/dit.pt \
    --classifier_checkpoint guided_diffusion/checkpoints/moment_classifier/checkpoint_best.pt \
    --target_smiles "c1ccccc1" \
    --guidance_scale 1.0 \
    --num_samples 100
```

---

## Detailed Usage

### 1. Direct Scattering Guidance

Uses differentiable scattering transform to compute gradients on the model's predicted clean graph.

```python
from guided_diffusion.generator import GuidedGraphDIT

# Load model
model = GuidedGraphDIT()
model.load_from_local("dit_checkpoint.pt")

# Generate with guidance
smiles = model.guided_generate(
    target_moments=target_moments,  # [D] or [B, D] tensor
    num_nodes=10,
    batch_size=32,
    guidance_scale=1.0,
    guidance_start_step=0,      # When to start guiding
    guidance_end_step=450,      # When to stop (None = guide until end)
)
```

### 2. Classifier Guidance

Uses a trained classifier to predict moments from noisy graphs.

#### Step 1: Prepare Training Data

```python
from guided_diffusion.prepare_classifier_data import prepare_classifier_dataset

prepare_classifier_dataset(
    data_dir="grassy_dit/data",
    output_dir="guided_diffusion/classifier_data",
    num_timesteps_per_mol=10,  # Sample 10 timesteps per molecule
    max_samples=10000,         # Limit dataset size
)
```

This generates `(noisy_X, noisy_E, t, node_mask, clean_moments)` tuples.

#### Step 2: Train Classifier

```python
from guided_diffusion.train_classifier import train_classifier

train_classifier(
    data_path="guided_diffusion/classifier_data/classifier_training_data.pt",
    output_dir="guided_diffusion/checkpoints/moment_classifier",
    hidden_size=256,
    num_layers=4,
    epochs=100,
    init_from_dit="path/to/dit.pt",  # Optional: transfer learning
    freeze_encoder=False,             # Optional: train only output head
)
```

#### Step 3: Generate

```python
from guided_diffusion.classifier_generator import ClassifierGuidedGraphDIT

model = ClassifierGuidedGraphDIT()
model.load_from_local("dit_checkpoint.pt")

smiles = model.classifier_guided_generate(
    classifier_path="guided_diffusion/checkpoints/moment_classifier/checkpoint_best.pt",
    target_moments=target_moments,
    num_nodes=10,
    batch_size=32,
    guidance_scale=1.0,
)
```

### 3. Evaluation

Compare guidance approaches:

```bash
python -m guided_diffusion.evaluate_guidance \
    --dit_checkpoint path/to/dit.pt \
    --classifier_checkpoint path/to/classifier.pt \
    --test_smiles_file test_molecules.txt \
    --num_samples_per_target 10 \
    --guidance_scales 0.1 0.5 1.0 2.0 \
    --output_dir evaluation_results
```

Output metrics:
- **Validity**: Fraction of valid molecules
- **Uniqueness**: Fraction of unique valid molecules  
- **Diversity**: 1 - average pairwise Tanimoto similarity
- **Moment MSE**: Distance to target scattering moments

---

## Key Parameters

### Guidance Scale

Controls the strength of guidance. Higher values push harder toward target moments but may reduce validity.

| Scale | Effect |
|-------|--------|
| 0.0 | No guidance (unguided generation) |
| 0.1-0.5 | Light guidance, high validity |
| 1.0 | Moderate guidance (recommended starting point) |
| 2.0+ | Strong guidance, may reduce validity |

### Guidance Window

Control when guidance is applied during diffusion:

```python
model.guided_generate(
    guidance_start_step=0,    # Start guiding from beginning
    guidance_end_step=450,    # Stop guiding at step 450 (of 500)
)
```

**Tip**: Stopping guidance early (e.g., last 10% of steps) allows the model to "clean up" invalid chemistry without guidance interference.

---

## Classifier Architecture

The `MomentClassifier` is a GNN regressor:

```
Input: (X_t, E_t, t, node_mask)
    │
    ▼
┌─────────────────────────────────┐
│ Input Embedding                  │  X + flatten(E) → Linear → hidden_size
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Timestep Embedding               │  Sinusoidal → MLP → hidden_size
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Transformer Blocks (×N)          │  Self-Attention + MLP with AdaLN modulation
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Global Pooling                   │  Masked mean over nodes
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│ Output MLP                       │  hidden_size → moment_dim
└─────────────────────────────────┘
    │
    ▼
Output: predicted_moments [B, moment_dim]
```

### Transfer Learning from DiT

Initialize classifier weights from a trained DiT:

```bash
python -m guided_diffusion.train_classifier \
    --data_path classifier_data.pt \
    --init_from_dit path/to/dit.pt \
    --freeze_encoder  # Optional: only train output head
```

This transfers the graph transformer layers that have learned to process noisy molecular graphs.

---

## Comparison: Direct vs Classifier Guidance

| Aspect | Direct Scattering | Classifier |
|--------|-------------------|------------|
| **Operates on** | Predicted clean graph | Current noisy state |
| **Requires training** | No | Yes (classifier) |
| **Gradient source** | Differentiable scattering | Learned classifier |
| **Computation** | Forward + backward through scattering | Forward + backward through classifier |
| **Flexibility** | Fixed scattering transform | Can learn data-specific patterns |

### When to Use Which

**Direct Scattering Guidance**:
- Quick experiments without training a classifier
- When scattering transform exactly matches your conditioning
- Smaller datasets where classifier might overfit

**Classifier Guidance**:
- Large-scale generation with pre-trained classifier
- When classifier can learn dataset-specific patterns
- When you want to experiment with different classifier architectures

---

## Troubleshooting

### Low Validity with High Guidance Scale

Try:
1. Reduce guidance scale (start with 0.1-0.5)
2. Use guidance window (`guidance_end_step` < total steps)
3. Combine with classifier-free guidance (`guide_scale` parameter)

### Classifier Not Improving Loss

Check:
1. Per-timestep bucket losses (early/mid/late should all decrease)
2. Try initializing from DiT checkpoint
3. Increase model capacity or training epochs

### Memory Issues

- Reduce batch size in data preparation and training
- Use gradient checkpointing for large classifiers
- Process molecules in smaller batches during generation

---

## References

- **Classifier Guidance**: Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis" (2021)
- **Graph DiT**: torch-molecule library
- **Scattering Transform**: Mallat, "Group Invariant Scattering" (2012)
