#### GRASSY-Net

A novel representation-first approach to molecular graph generation.

#### GRASSY-Net — Usage: data preparation → learnable scattering → GRASSY training → latent visualization

### Environment Creation

```
uv venv
uv sync
uv pip install torch_geometric
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```
Prereqs
- Python 3.8+
- numpy, pandas, scikit-learn (or umap-learn), matplotlib
- repository scripts: train_learnable_scattering.py, train_grassy.py (expecting the file arguments below)
Datasets helper scripts

1) Convert CSV to .npy and compute relevant properties of the molecules.

    - datasets/csv_to_npy.py — convert CSV (columns: "smiles", prop1, prop2, ...) → single container .npy (dict with "smiles", "props", "prop_names") or optional separate smiles/props .npy
        - example:
            ```
            python datasets/csv_to_npy.py --input molecules.csv --output molecules.npy
            ```
        - output: molecules.npy (load later with np.load(..., allow_pickle=True).item() if saved as a dict)
2) Compute per-property statistics

    - datasets/compute_tranche_statistics.py — compute per-property mean/std/min/max given the molecules.npy file computed from step 1.
        - example:
            ```
            python datasets/compute_tranche_statistics.py
            ```
        - outputs: molecules_stats.npy

Notes
- Use these scripts to keep preprocessing reproducible; they replace the inline CSV->npy and stats code snippets above.
- Preserve molecule ordering across all steps (CSV → scattering → GRASSY). If you filter/reorder, save an index mapping file.
- When loading molecules.npy that contains a dict, use np.load(..., allow_pickle=True).item().
- If your filenames differ, adjust the CLI arguments accordingly.

3) Load data into train_learnable_scattering
- Expected inputs: molecules.npy and statistics file molecules_stats.npy
- Uses the dataloader defined in datasets/load_ZINC_tranche.py
- CLI example:

```
python train_learnable_scattering.py
```

- Loading inside a script:

```python
dataset = ZINCDataset(f'datasets/fields_1.npy', prop_stat_dict=f'datasets/fields_1_stats.npy')
# pass smiles/props to dataset creation used by train_learnable_scattering
```

- Output: trained weights + scattering moments file, e.g. scripts/trained_models/molecules.npy

4) Load scattering moments into train_grassy
- Ensure scattering moments align with the same molecule order used for properties.
- Expected inputs: molecules.npy, statistics file molecules_stats.npy, and saved scattering model scripts/trained_models/molecules.npy
- CLI example:

```
python train_grassy.py
```

- In-code example:

```python
full_dataset = ZINCDataset(f'datasets/molecules.npy', prop_stat_dict=f'datasets/molecules_stats.npy', transform=Scattering(scatter_model_name=f'scripts/trained_models/molecules.npy'))
```

- Output: GRASSY model + latent embeddings file, e.g. outputs/grassy/embeddings.npy (shape: N x D)

5) Visualize latent embeddings colored by a property and save visualize_latent_embeddings.npy
- Load embeddings and pick a property (by name or index).


Notes & tips
- Keep molecule order consistent across all steps (CSV → scattering → GRASSY). If any data filtering/reordering occurs, store an index mapping file.
- Use prop_mean.npy/prop_std.npy to normalize properties before training.
- Verify shapes at each step (N molecules) and check for NaNs.
- If your training scripts accept .npz or separate .npy files, adapt the CLI accordingly.

This sequence produces:
- molecules.npy (raw data)
- property_stats.json / prop_mean.npy / prop_std.npy
- outputs/learnable_scattering/scattering_moments.npy
- outputs/grassy/embeddings.npy
- visualize_latent_embeddings.npy (2D projection for plotting)
- corresponding visualization PNG(s)