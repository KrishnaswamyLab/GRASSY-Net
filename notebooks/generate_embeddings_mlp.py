#!/usr/bin/env python3
"""
GRASSY Latent Space Visualization

This script loads a trained GRASSY model and generates embeddings for visualization.
Converted from notebook: 1_0-generate_embeddings_mlp.ipynb
"""

import os
import sys
import argparse
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import project-specific modules after path setup
from models.GRASSY_model import GRASSY
from models.MLP_ScatteringTransform import GraphScatteringTransform
from datasets.ZINCDataset import ZINCDataset
from models.EndToEndWrapper import EndToEndScatteringGRASSYWrapper
from utils.config_utils import config_to_hparams, load_config, get_grassy_flags

# Usage:
# python -m notebooks.generate_embeddings_mlp --save_dir outputs/MOSES_12K_e2e_regress_nokld_2026-01-21-18-20-29 --model_path outputs/MOSES_12K_e2e_regress_nokld_2026-01-21-18-20-29/best-epoch=74-val_loss=0.005.ckpt

class ScatteringDataset(torch.utils.data.Dataset):
    """Wrapper dataset that applies scattering transform."""
    
    def __init__(self, base_dataset, scattering_transform):
        self.base_dataset = base_dataset
        self.scattering_transform = scattering_transform
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        data = self.base_dataset[idx]
        with torch.no_grad():
            scat_coeffs = self.scattering_transform(data)
        return scat_coeffs.squeeze(), data.y  # Add .squeeze() to remove extra dimensions


def create_model(config, scattering_dim, num_properties, device):
    """Create and return the GRASSY model."""
    training_cfg = config['training']
    grassy_version = training_cfg['grassy_version']
    kl_div, reg = get_grassy_flags(grassy_version)
    
    alpha = training_cfg['alpha'] if reg else 0
    beta = training_cfg['beta'] if kl_div else 0
    
    len_epoch = 1  # Not needed for inference
    hparams = config_to_hparams(config, scattering_dim, num_properties, len_epoch)
    hparams.alpha = alpha
    hparams.beta = beta
    
    grassy_model = GRASSY(hparams=hparams)
    
    return grassy_model, hparams, alpha, beta


def load_model_weights(model, model_path, device):
    """Load trained weights into model."""
    if model_path.endswith('.ckpt'):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Loaded checkpoint from: {model_path}")
    else:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded state dict from: {model_path}")
    
    model.to(device)
    model.eval()
    print(f"Model loaded and moved to {device}")
    return model


def generate_scattering_coefficients(full_dataset, no_transform_dataset, num_classes):
    """Generate scattering coefficients and collect properties."""
    print("Generating scattering coefficients for full dataset...")
    
    scat_mom_list = []
    prop = [[] for _ in range(num_classes)]
    
    # Atom percentages
    carbon = []
    nitro = []
    oxy = []
    
    for index in tqdm(range(len(full_dataset))):
        entry = full_dataset[index]
        scat_mom_list.append(entry[0].detach().cpu().numpy())
        
        # Save all properties
        # Add this debug print before the loop in generate_scattering_coefficients
        entry = full_dataset[0]
        y = entry[1].detach().cpu().numpy().flatten()  # Convert entire tensor to numpy and flatten it to be (num_classes,)
        for i in range(num_classes):
            prop[i].append(y[i])

        # Get atom counts from original dataset
        data = no_transform_dataset[index]
        
        c, n, o = 0, 0, 0
        atom_count = 0
        
        for atom in data.element:
            if atom == 'C':
                c += 1
            elif atom == 'N':
                n += 1
            elif atom == 'O':
                o += 1
            atom_count += 1
        
        carbon.append(c / atom_count if atom_count > 0 else 0)
        nitro.append(n / atom_count if atom_count > 0 else 0)
        oxy.append(o / atom_count if atom_count > 0 else 0)
    
    scat_mom_list = np.array(scat_mom_list)
    print(f"Scattering coefficients shape: {scat_mom_list.shape}")
    
    return scat_mom_list, prop, carbon, nitro, oxy


def generate_embeddings(model, scat_mom_list, device):
    """Generate embeddings through the model."""
    moments = torch.Tensor(scat_mom_list).to(device)
    
    with torch.no_grad():
        ordered_embed = model.grassy.embed(moments)[0]
    
    ordered_embed_np = ordered_embed.cpu().numpy()
    print(f"Embedding shape: {ordered_embed_np.shape}")
    return ordered_embed_np


def save_embeddings(output_dir, prefix, ordered_embed_np, scat_mom_list, prop, atom_percentage):
    """Save embeddings to disk."""
    np.save(os.path.join(output_dir, f"ordered_embedding_{prefix}.npy"), ordered_embed_np)
    np.save(os.path.join(output_dir, f"scattering_coeffs_{prefix}.npy"), scat_mom_list)
    np.save(os.path.join(output_dir, f"embedding_prop_lists_{prefix}.npy"), np.array(prop, dtype=object))
    np.save(os.path.join(output_dir, f"atom_percentages_{prefix}.npy"), np.array(atom_percentage))
    print(f"Embeddings saved to {output_dir}")


def main(args):
    
    # Configuration
    save_dir = args.save_dir
    config_path = os.path.join(save_dir, 'config.yaml')
    model_path = args.model_path or os.path.join(save_dir, 'best-epoch=93-val_loss=0.006.ckpt')
    
    # Load config
    config = load_config(config_path)
    dataset_cfg = config['dataset']
    model_cfg = config['model']
    training_cfg = config['training']
    scattering_cfg = config['scattering']
    
    print("Config loaded successfully!")
    print(f"Dataset: {dataset_cfg['name']}")
    print(f"GRASSY version: {training_cfg['grassy_version']}")
    
    # Load dataset
    no_transform_dataset = ZINCDataset(dataset_cfg['path'])
    base_dataset = ZINCDataset(
        dataset_cfg['path'],
        prop_stat_dict=dataset_cfg.get('stats_path'),
        transform=None,
        include_ki=dataset_cfg.get('include_ki', False)
    )
    
    print(f"Loaded {len(base_dataset)} molecules")
    print(f"Node features: {base_dataset.num_node_features}")
    print("In Channels for Scattering Transform:", scattering_cfg['in_channels'])
    print(f"Properties: {base_dataset.num_classes}")
    
    # Create scattering transform
    scattering_transform = GraphScatteringTransform(
        in_channels=scattering_cfg['in_channels'],
        J=scattering_cfg['J'],
        num_moments=scattering_cfg['num_moments'],
        mlp_hidden_dim=scattering_cfg.get('mlp_hidden_dim', 64),
    )
    
    scattering_dim = scattering_transform.out_shape()
    print(f"Scattering output dimension: {scattering_dim}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grassy_model, hparams, alpha, beta = create_model(
        config, scattering_dim, base_dataset.num_classes, device
    )
    
    model = EndToEndScatteringGRASSYWrapper(
        scattering_transform, grassy_model, hparams, alpha, beta
    )
    print("Model architecture created!")
    
    # Load weights
    model = load_model_weights(model, model_path, device)
    
    # Create dataset with scattering transform
    full_dataset = ScatteringDataset(base_dataset, scattering_transform)
    print(f"Full dataset size: {len(full_dataset)}")
    
    # Generate scattering coefficients
    scat_mom_list, prop, carbon, nitro, oxy = generate_scattering_coefficients(
        full_dataset, no_transform_dataset, base_dataset.num_classes
    )
    
    # Generate embeddings
    ordered_embed_np = generate_embeddings(model, scat_mom_list, device)
    
    # Save embeddings
    output_dir = args.output_dir or save_dir
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{dataset_cfg['name']}_e2e"
    atom_percentage = [carbon, nitro, oxy]
    
    if args.save_embeddings:
        print("Saving embeddings to disk...")
        save_embeddings(output_dir, prefix, ordered_embed_np, scat_mom_list, prop, atom_percentage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate GRASSY embeddings for visualization")
    
    parser.add_argument(
        "--save_dir", 
        type=str, 
        required=True,
        help="Path to training output directory containing config.yaml"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default=None,
        help="Path to model checkpoint (.ckpt or .pt file)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Output directory for embeddings and visualizations (default: same as save_dir)"
    )

    parser.add_argument(
        "--save_embeddings", 
        action="store_true",
        help="Save embeddings to disk"
    )
 
    args = parser.parse_args()
    main(args)