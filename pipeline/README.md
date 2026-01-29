# GRASSY-Net End-to-End Pipeline

A unified pipeline that automates the complete GRASSY-Net workflow: from raw SMILES data to trained models and evaluation reports.

## Overview

The pipeline orchestrates 6 stages:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Prep      │ -> │  Scattering     │ -> │  Splitting      │
│  (SMILES→Graphs)│    │  (Extract GST)  │    │  (Train/Val/Test)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
         ┌────────────────────────────────────────────┘
         ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Train GRASSY   │ -> │  Train DiT      │ -> │  Evaluate       │
│  (Autoencoder)  │    │  (Diffusion)    │    │  (Metrics/Report)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Quick Start

```bash
# Run full pipeline from scratch
python -m pipeline.run_pipeline --input datasets/ZINC_tranches/BBAB/BBAB.smi

# Use existing checkpoints (skip training)
python -m pipeline.run_pipeline --input data.smi \
    --grassy-checkpoint path/to/grassy.ckpt --grassy-epochs 0 \
    --dit-checkpoint path/to/dit.pt --dit-epochs 0

# Only train DiT (use existing GRASSY)
python -m pipeline.run_pipeline --input data.smi \
    --grassy-checkpoint path/to/grassy.ckpt --grassy-epochs 0
```

## CLI Reference

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--input`, `-i` | Input `.smi` file with SMILES strings |

### Output Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir`, `-o` | `runs/{name}_{timestamp}` | Output directory |
| `--name` | `experiment` | Experiment name |
| `--config`, `-c` | None | YAML config file |

### Data Splitting

| Argument | Default | Description |
|----------|---------|-------------|
| `--train-ratio` | 0.8 | Training set ratio |
| `--val-ratio` | 0.1 | Validation set ratio |
| `--test-ratio` | 0.1 | Test set ratio |
| `--split-seed` | 42 | Random seed for splitting (reproducible) |

> **Note:** The split seed only affects data partitioning. Training and sampling remain stochastic.

### GRASSY Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--grassy-checkpoint` | None | Path to existing GRASSY checkpoint |
| `--grassy-epochs` | 100 | Training epochs (0 = skip training) |

### DiT Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--dit-checkpoint` | None | Path to existing DiT checkpoint |
| `--dit-epochs` | 2000 | Training epochs (0 = skip training) |
| `--noise-prob` | 0.2 | Probability of adding noise during training |
| `--noise-std` | 0.2 | Gaussian noise standard deviation |

### Evaluation

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-samples` | 1000 | Molecules to generate for evaluation |

### Hardware & Logging

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | `auto` | Device: `auto`, `cuda`, or `cpu` |
| `--wandb` | disabled | Enable Weights & Biases logging |
| `--no-wandb` | - | Explicitly disable W&B |

### Advanced

| Argument | Description |
|----------|-------------|
| `--override` | Override config values (e.g., `dit.hidden_size=512`) |

## YAML Configuration

For complex experiments, use a YAML config file:

```yaml
# my_config.yaml
experiment:
  name: "BBAB_experiment"
  output_dir: "runs/bbab_2026"

dataset:
  input_file: "datasets/ZINC_tranches/BBAB/BBAB.smi"
  max_molecules: null  # null = use all

splitting:
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
  seed: 42

scattering:
  J: 4
  num_moments: 4

grassy:
  checkpoint: null
  epochs: 100
  bottle_dim: 32
  hidden_dim: 128
  batch_size: 64
  learning_rate: 0.001
  alpha: 0.01

dit:
  checkpoint: null
  epochs: 2000
  hidden_size: 1024
  num_layer: 6
  num_head: 16
  batch_size: 64
  learning_rate: 2e-4
  noise_prob: 0.2
  noise_std: 0.2

evaluation:
  num_samples: 1000
  batch_size: 64
  guide_scale: 2.0

hardware:
  device: "auto"
  num_workers: 4

logging:
  wandb:
    enabled: false
    project: "GRASSY-Pipeline"
    entity: "grassy"
```

Run with:

```bash
python -m pipeline.run_pipeline --config my_config.yaml
```

CLI arguments override YAML values:

```bash
python -m pipeline.run_pipeline --config my_config.yaml --dit-epochs 500
```

## Common Workflows

### 1. Full Training from Scratch

```bash
python -m pipeline.run_pipeline \
    --input datasets/ZINC_tranches/BBAB/BBAB.smi \
    --name bbab_full \
    --grassy-epochs 100 \
    --dit-epochs 2000 \
    --wandb
```

### 2. Train Only DiT (Existing GRASSY)

```bash
python -m pipeline.run_pipeline \
    --input datasets/ZINC_tranches/BBAB/BBAB.smi \
    --grassy-checkpoint outputs/BBAB_grassy/best.ckpt \
    --grassy-epochs 0 \
    --dit-epochs 2000
```

### 3. Evaluate Existing Models

```bash
python -m pipeline.run_pipeline \
    --input datasets/ZINC_tranches/BBAB/BBAB.smi \
    --grassy-checkpoint outputs/grassy.ckpt \
    --grassy-epochs 0 \
    --dit-checkpoint outputs/dit.pt \
    --dit-epochs 0 \
    --num-samples 5000
```

### 4. Custom Split for Cross-Validation

```bash
# Run 1 - seed 42
python -m pipeline.run_pipeline --input data.smi --split-seed 42 --output-dir runs/cv_fold1

# Run 2 - seed 123
python -m pipeline.run_pipeline --input data.smi --split-seed 123 --output-dir runs/cv_fold2
```

### 5. Quick Test Run

```bash
python -m pipeline.run_pipeline \
    --input datasets/ZINC_tranches/BBAB/BBAB.smi \
    --grassy-epochs 2 \
    --dit-epochs 10 \
    --num-samples 50
```

## Output Structure

```
runs/bbab_experiment_20260129/
├── pipeline_config.yaml      # Saved configuration
├── pipeline_metadata.json    # Stage tracking & provenance
├── final_report.md           # Human-readable report
├── final_report.json         # Machine-readable results
│
├── data_prep/                # Stage 1: Processed molecules
│   ├── BBAB_processed.pkl
│   └── BBAB_stats.pkl
│
├── scattering/               # Stage 2: Scattering moments
│   ├── scattering_moments.npy
│   └── molecules.csv
│
├── splitting/                # Stage 3: Train/Val/Test splits
│   ├── train/
│   ├── val/
│   └── test/
│
├── train_grassy/             # Stage 4: GRASSY model
│   └── best.ckpt
│
├── train_dit/                # Stage 5: DiT model
│   ├── model.pt
│   └── dit_config.yaml
│
└── evaluate/                 # Stage 6: Evaluation results
    ├── generated_molecules.csv
    ├── metrics.json
    └── plots/
```

## Checkpoint Injection

The pipeline supports injecting pre-trained checkpoints at any stage:

- **Setting `--*-epochs 0`** with a checkpoint → Skips training, uses provided checkpoint
- **Setting `--*-epochs N`** with a checkpoint → Continues training from checkpoint
- **Setting `--*-epochs N`** without checkpoint → Trains from scratch

This enables:
- Reusing expensive GRASSY training across experiments
- Ablation studies with different DiT configurations
- Evaluation of external checkpoints

## Architecture

```
pipeline/
├── __init__.py              # Public API
├── run_pipeline.py          # Main CLI entry point
├── pipeline_config.py       # Configuration schema (Pydantic)
├── checkpoint_manager.py    # Output/checkpoint tracking
├── report_generator.py      # Final report generation
│
├── configs/
│   └── default_config.yaml  # Template configuration
│
└── stages/                  # Stage wrappers (don't modify originals)
    ├── __init__.py
    ├── data_prep.py         # → datasets/prepare_zinc_tranche.py
    ├── scattering.py        # → grassy_dit/extract_scattering_fixed.py
    ├── splitting.py         # → grassy_dit/split_datasets.py
    ├── train_grassy.py      # → train_grassy_fixed_scattering.py
    ├── train_dit.py         # → grassy_dit/train.py
    └── evaluate.py          # → evaluation/zinc_eval_direct.py
```

## Notes

- **No original files modified**: All stage wrappers call existing scripts without modification
- **Reproducible splits**: Use `--split-seed` for deterministic data partitioning
- **Stochastic training**: Training and sampling do NOT use fixed seeds (intentional)
- **GPU auto-detection**: Set `--device auto` (default) for automatic GPU usage
