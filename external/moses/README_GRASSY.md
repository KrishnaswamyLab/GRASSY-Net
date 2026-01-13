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

### 2. Data Files

The dataset files (`train.csv.gz`, `test.csv.gz`, `test_scaffolds.csv.gz`) are stored via Git LFS in the original repo. For this fork, download them manually:

```bash
cd moses/dataset/data
curl -L -o train.csv.gz "https://media.githubusercontent.com/media/molecularsets/moses/master/moses/dataset/data/train.csv.gz"
curl -L -o test.csv.gz "https://media.githubusercontent.com/media/molecularsets/moses/master/moses/dataset/data/test.csv.gz"
curl -L -o test_scaffolds.csv.gz "https://media.githubusercontent.com/media/molecularsets/moses/master/moses/dataset/data/test_scaffolds.csv.gz"
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
| Novelty | Fraction not in training set |

## Original Repository

https://github.com/molecularsets/moses
