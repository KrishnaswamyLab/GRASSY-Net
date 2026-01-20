# MOSES Benchmark Evaluation for GRASSY-DiT

This module provides tools to evaluate GRASSY-DiT molecular generation against MOSES baselines (VAE, AAE, CharRNN, JTN, etc.) and DiGress.

## Four Evaluation Modes

| Mode | Pipeline | Use Case |
|------|----------|----------|
| `unconditional` | DiT generates randomly (no conditioning) | Baseline comparison |
| `direct` | molecule → scattering → DiT | Standard conditioning |
| `full-pipeline` | molecule → scattering → GRASSY VAE → moments → DiT | With VAE round-trip |
| `latent-sample` | random latent z → GRASSY decode → moments → DiT | Novel molecule exploration |

## Directory Structure

```
evaluation/
├── __init__.py              # Module exports
├── moses_benchmark.py       # Main evaluation script
├── utils.py                 # Shared utilities (model loading, scattering)
├── load_baselines.py        # Load MOSES/DiGress baseline samples
├── checkpoints/             # External model checkpoints
│   ├── digress/             # Place DiGress checkpoints here
│   └── README.md            # Checkpoint instructions
├── results/                 # Output directory for results
└── README.md                # This file
```

## Usage Examples

### Mode 0: Unconditional Generation

Generate molecules without any conditioning (baseline comparison):

```bash
python -m evaluation.moses_benchmark \
    --dit-checkpoint grassy_dit_checkpoint.pt \
    --mode unconditional \
    --num-samples 10000
```

### Mode 1: Direct Scattering Conditioning (Default)

Condition DiT directly on scattering moments from source molecules:

```bash
python -m evaluation.moses_benchmark \
    --dit-checkpoint grassy_dit_checkpoint.pt \
    --mode direct \
    --source train \
    --num-samples 10000
```

### Mode 2: Full Pipeline with GRASSY VAE Round-Trip

Route scattering through GRASSY VAE encode/decode before conditioning:

```bash
python -m evaluation.moses_benchmark \
    --dit-checkpoint grassy_dit_checkpoint.pt \
    --grassy-checkpoint path/to/grassy_vae.pt \
    --mode full-pipeline \
    --num-samples 10000
```

### Mode 3: Sample from GRASSY Latent Space

Generate novel molecules by sampling random points from the GRASSY latent space:

```bash
python -m evaluation.moses_benchmark \
    --dit-checkpoint grassy_dit_checkpoint.pt \
    --grassy-checkpoint path/to/grassy_vae.pt \
    --mode latent-sample \
    --num-samples 10000
```

### Compare with DiGress

Place a DiGress checkpoint in `evaluation/checkpoints/digress/` and run:

```bash
python -m evaluation.moses_benchmark \
    --dit-checkpoint grassy_dit_checkpoint.pt \
    --compare-digress \
    --num-samples 10000
```

### Use Learnable Scattering

Use a trained learnable scattering model instead of fixed:

```bash
python -m evaluation.moses_benchmark \
    --dit-checkpoint grassy_dit_checkpoint.pt \
    --scattering-checkpoint path/to/learnable_scatter.pt \
    --mode direct
```

### List Available Baselines

Check which baseline models have samples available:

```bash
python -m evaluation.moses_benchmark --list-baselines
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dit-checkpoint` | required | Path to trained GRASSY-DiT model |
| `--grassy-checkpoint` | None | Path to GRASSY VAE (required for `full-pipeline` and `latent-sample`) |
| `--scattering-checkpoint` | None | Path to learnable scattering (uses fixed `GraphScatteringTransform` if None) |
| `--mode` | `direct` | One of: `unconditional`, `direct`, `full-pipeline`, `latent-sample` |
| `--source` | `train` | `train` (random training mols) or `test` (reconstruction task) |
| `--num-samples` | 10000 | Number of molecules to generate |
| `--batch-size` | 64 | Generation batch size |
| `--compare-baselines` | True | Compare against MOSES pre-generated baselines |
| `--compare-digress` | False | Compare against DiGress |
| `--output-dir` | `evaluation/results/` | Output directory for results |
| `--device` | `cuda` | Computation device (`cuda` or `cpu`) |
| `--n-jobs` | 4 | Number of workers for metric computation |

## MOSES Metrics

| Metric | Description |
|--------|-------------|
| valid | Fraction of valid SMILES |
| unique@1k / unique@10k | Fraction of unique molecules |
| FCD/Test | Fréchet ChemNet Distance |
| SNN/Test | Similarity to nearest neighbor |
| Frag/Test | Fragment similarity |
| Scaf/Test | Scaffold similarity |
| IntDiv / IntDiv2 | Internal diversity |
| Filters | Medicinal chemistry filters pass rate |
| Novelty | Fraction not in training set |

## Output Format

Results are saved as JSON in `evaluation/results/`:

```json
{
  "config": {
    "mode": "direct",
    "source": "train",
    "num_samples": 10000,
    "timestamp": "20260120_143022"
  },
  "grassy_dit": {
    "valid": 0.95,
    "unique@10k": 0.98,
    "FCD/Test": 0.45,
    "Novelty": 0.92
  },
  "baselines": {
    "vae": {"valid": 0.977, "unique@10k": 0.998},
    "aae": {"valid": 0.937},
    "char_rnn": {"valid": 0.975}
  },
  "comparison_table": "| Metric | GRASSY-DiT | vae | ... |"
}
```

Generated samples are also saved as `grassy_dit_samples_{mode}_{timestamp}.txt`.

## Programmatic Usage

```python
from evaluation import MOSESBenchmark, run_benchmark

# Option 1: Using the class
benchmark = MOSESBenchmark(
    dit_checkpoint="grassy_dit_checkpoint.pt",
    grassy_checkpoint="grassy_vae.pt",  # optional
    device="cuda",
)

results = benchmark.run(
    mode="direct",
    num_samples=10000,
    compare_baselines=True,
)

# Option 2: Using the convenience function
results = run_benchmark(
    dit_checkpoint="grassy_dit_checkpoint.pt",
    mode="direct",
    num_samples=10000,
)
```

## MOSES Baselines

The following pre-generated baselines are available in `external/moses/data/samples/`:

- VAE
- AAE
- CharRNN
- JTN (Junction Tree)
- LatentGAN
- NGram
- HMM
- Combinatorial

## Adding DiGress for Comparison

See `evaluation/checkpoints/README.md` for instructions on adding DiGress checkpoints.
