"""Debug helper: inspect first N items of ZINCDataset and print attribute types/dtypes.

Run from repo root:
    python scripts/debug_dataset.py
"""
from datasets.ZINCDataset import ZINCDataset
import torch

def inspect(tranch_file, n=10):
    ds = ZINCDataset(tranch_file)
    print(f"Dataset length: {len(ds)}")
    for i in range(min(n, len(ds))):
        try:
            data = ds[i]
        except Exception as e:
            print(f"Error reading index {i}: {e}")
            continue
        print(f"--- Item {i} ---")
        for k in data.keys:
            v = data[k]
            if isinstance(v, torch.Tensor):
                print(f"{k}: Tensor shape={tuple(v.shape)} dtype={v.dtype}")
            else:
                print(f"{k}: {type(v)} value_sample={v if (isinstance(v, (int,float,str)) or v is None) else type(v)}")

if __name__ == '__main__':
    # default tranche used in train_learnable_scattering.py
    TRANCH = 'BBAB_subset.npy'
    import os
    p = os.path.join('datasets', TRANCH)
    inspect(p, n=20)
