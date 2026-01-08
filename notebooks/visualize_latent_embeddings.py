#!/usr/bin/env python3
"""
visualize_latent_embeddings.py

Visualize latent embeddings (e.g., from ProtSCAPE or HiPoNet) using PCA and PHATE,
colored by one or more properties.

Usage:
    python visualize_latent_embeddings.py \
        --embedding path/to/embedding.npy \
        --labels path/to/labels.npy \
        [--outdir figs] \
        [--no-show]
"""

import os
import argparse
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import phate

def load_embedding(path):
    x = np.load(path)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    return x

def compute_pca(X, n_components=2):
    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X)
    return Z, pca

def compute_phate(X, n_components=2, knn=5, decay=40):
    if phate is None:
        raise ImportError("phate is not installed. Install with: pip install phate")
    phate_op = phate.PHATE(n_components=n_components, knn=knn, decay=decay)
    Z = phate_op.fit_transform(X)
    return Z, phate_op

def plot_scatter(Z, colors=None, title="", save_path=None, s=10, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=colors, s=s, cmap=cmap)
    plt.colorbar(sc, ax=ax, label="Property value")
    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✅ Saved figure: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", required=True, help="Path to .npy embedding file")
    parser.add_argument("--labels", default=None, help="Optional .npy file with labels or values to color by (shape = (num_properties, num_samples))")
    parser.add_argument("--outdir", default="figs", help="Directory to save figures")
    parser.add_argument("--no-show", action="store_true", help="Do not show figures interactively")
    parser.add_argument("--phate-knn", type=int, default=5, help="PHATE knn parameter")
    parser.add_argument("--phate-decay", type=float, default=40.0, help="PHATE decay parameter")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading embedding:", args.embedding)
    X = load_embedding(args.embedding)
    print("Embedding shape:", X.shape)

    labels = None
    if args.labels:
        print("Loading labels:", args.labels)
        lbl = np.load(args.labels)
        lbl = np.atleast_2d(lbl)  

        if lbl.shape[0] < lbl.shape[1]:
            lbl = lbl.T
        if lbl.shape[0] != X.shape[0]:
            print(f"Warning: Labels length ({lbl.shape[0]}) does not match embeddings ({X.shape[0]}). Ignoring labels.")
        else:
            labels = lbl
            print(f"Labels shape: {labels.shape}")

    # ---- PCA ----
    print("Computing PCA...")
    Z_pca, pca_model = compute_pca(X, n_components=2)
    print("Explained variance ratio (PCA):", pca_model.explained_variance_ratio_)

    # ---- PHATE ----
    phate_available = phate is not None
    if phate_available:
        print("Computing PHATE...")
        Z_phate, phate_model = compute_phate(X, n_components=2, knn=args.phate_knn, decay=args.phate_decay)
    else:
        print("PHATE not available. To enable, run: pip install phate")
        Z_phate = None

    # ---- Plotting ----
    if labels is not None:
        num_props = labels.shape[1]
        for i in range(num_props):
            prop_vals = labels[:, i]
            plot_scatter(
                Z_pca, colors=prop_vals,
                title=f"PCA (Property {i+1})",
                save_path=os.path.join(args.outdir, f"latent_pca_prop{i+1}.png")
            )
            if phate_available:
                plot_scatter(
                    Z_phate, colors=prop_vals,
                    title=f"PHATE (Property {i+1})",
                    save_path=os.path.join(args.outdir, f"latent_phate_prop{i+1}.png")
                )
    else:
        # single unlabeled plots
        plot_scatter(Z_pca, title="PCA (Unlabeled)", save_path=os.path.join(args.outdir, "latent_pca.png"))
        if phate_available:
            plot_scatter(Z_phate, title="PHATE (Unlabeled)", save_path=os.path.join(args.outdir, "latent_phate.png"))

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
