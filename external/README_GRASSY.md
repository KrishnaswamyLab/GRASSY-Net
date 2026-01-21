# MOSES Benchmark - GRASSY-Net Fork

This is a modified copy of the [MOSES benchmark](https://github.com/molecularsets/moses) integrated into GRASSY-Net for molecular generation evaluation.

## Changes from Original

### 1. Pandas 2.0+ Compatibility Fix (utils.py)

**File**: `moses/metrics/utils.py`, lines 23-26

**Problem**: Original code used `DataFrame.append()` which was removed in pandas 2.0.

**Fix**: Replaced with `pd.concat()`:
```python
# Original (broken in pandas 2.0+):
# _filters = [Chem.MolFromSmarts(x) for x in
#             _mcf.append(_pains, sort=True)['smarts'].values]

# Fixed:
_filters = [Chem.MolFromSmarts(x) for x in
            pd.concat([_mcf, _pains], sort=True)['smarts'].values]
```

### 2. NumPy Array Truth Value Fix (metrics.py)

**File**: `moses/metrics/metrics.py`, line 77

**Problem**: Original code used `train = train or get_dataset('train')` which fails with numpy arrays due to ambiguous truth values.

**Fix**: Changed to explicit None check:
```python
# Original (broken with numpy arrays):
# train = train or get_dataset('train')

# Fixed:
train = get_dataset('train') if train is None else train
```

### 3. Data Files

The dataset files (`train.csv.gz`, `test.csv.gz`, `test_scaffolds.csv.gz`, `test_stats.npz`, `test_scaffolds_stats.npz`) are stored via Git LFS in the original repo. For this fork, download them manually:

```bash
cd moses/dataset/data
curl -L -o train.csv.gz "https://media.githubusercontent.com/media/molecularsets/moses/master/moses/dataset/data/train.csv.gz"
curl -L -o test.csv.gz "https://media.githubusercontent.com/media/molecularsets/moses/master/moses/dataset/data/test.csv.gz"
curl -L -o test_scaffolds.csv.gz "https://media.githubusercontent.com/media/molecularsets/moses/master/moses/dataset/data/test_scaffolds.csv.gz"
curl -L -o test_stats.npz "https://media.githubusercontent.com/media/molecularsets/moses/master/moses/dataset/data/test_stats.npz"
curl -L -o test_scaffolds_stats.npz "https://media.githubusercontent.com/media/molecularsets/moses/master/moses/dataset/data/test_scaffolds_stats.npz"
```

## Installation

From the GRASSY-Net root directory:

```bash
pip install -e external/moses --no-deps
```

## Usage

```python
from moses import get_all_metrics, get_dataset

# Load datasets
train = get_dataset('train')
test = get_dataset('test')

# Your generated SMILES
generated = ["CCO", "CCCO", ...]

# Compute all MOSES metrics
metrics = get_all_metrics(generated, train=train, test=test)
```

## MOSES Metrics

| Metric | Description |
|--------|-------------|
| valid | Fraction of valid SMILES |
| unique@1k, unique@10k | Fraction of unique molecules |
| FCD/Test | Fréchet ChemNet Distance |
| SNN/Test | Similarity to nearest neighbor |
| Frag/Test | Fragment similarity |
| Scaf/Test | Scaffold similarity |
| IntDiv, IntDiv2 | Internal diversity |
| Filters | Fraction passing medicinal chemistry filters |
| logP, SA, QED, weight | Distribution differences |
| Novelty | Fraction not in training set |

## Original Repository

https://github.com/molecularsets/moses
