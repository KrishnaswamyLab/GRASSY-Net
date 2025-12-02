"""
GRASSY-DiT training using torch-molecule's GraphDIT infrastructure.
Replaces their property conditioning with our cross-attention to scattering tokens.
"""
import torch
import torch.nn.functional as F
from torch_molecule import GraphDITMolecularGenerator
from torch_molecule.generator.graph_dit.utils import PlaceHolder
from grassy_dit.model import ScatteringDenoiser


class ScatteringTransformerAdapter(torch.nn.Module):
    """Wraps ScatteringDenoiser to match Transformer.forward(noisy_data, unconditioned) signature."""
    
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser
    
    def forward(self, noisy_data, unconditioned):
        X_t = noisy_data['X_t'].float()
        E_t = noisy_data['E_t'].float()
        node_mask = noisy_data['node_mask']
        t = noisy_data['t']
        scattering = noisy_data['y_t']  # [B, 440] scattering passed as y
        
        X_pred, E_pred = self.denoiser(X_t, E_t, node_mask, t, scattering, uncond=unconditioned)
        return PlaceHolder(X=X_pred, E=E_pred, y=None).mask(node_mask)
    
    def compute_loss(self, noisy_data, true_X, true_E, lw_X, lw_E, unconditioned=False):
        pred = self.forward(noisy_data, unconditioned=unconditioned)
        
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))
        masked_pred_X = torch.reshape(pred.X, (-1, pred.X.size(-1)))
        masked_pred_E = torch.reshape(pred.E, (-1, pred.E.size(-1)))
        
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)
        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]
        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]
        
        loss_X = F.cross_entropy(flat_pred_X, torch.argmax(flat_true_X, dim=-1)) if true_X.numel() > 0 else 0.0
        loss_E = F.cross_entropy(flat_pred_E, torch.argmax(flat_true_E, dim=-1)) if true_E.numel() > 0 else 0.0
        loss = lw_X * loss_X + lw_E * loss_E
        return loss, loss_X, loss_E


class ScatteringGraphDIT(GraphDITMolecularGenerator):
    """GraphDIT with scattering moment conditioning via cross-attention."""
    
    def __init__(self, **kwargs):
        kwargs['input_dim_y'] = 440  # scattering moments dimension
        super().__init__(**kwargs)
    
    def _initialize_model(self, model_class, checkpoint=None):
        """Override to use ScatteringDenoiser instead of Transformer."""
        denoiser = ScatteringDenoiser(
            max_n_nodes=self.max_node,
            hidden_size=self.hidden_size,
            depth=self.num_layer,
            num_heads=self.num_head,
        )
        self.model = ScatteringTransformerAdapter(denoiser).to(self.device)
        return self.model


if __name__ == "__main__":
    import argparse
    import numpy as np
    import pandas as pd
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--csv_file', default='molecules.csv')
    parser.add_argument('--smiles_col', default='smiles')
    parser.add_argument('--scatter_file', default='scattering_moments.npy')
    parser.add_argument('--max_node', type=int, default=50)
    parser.add_argument('--hidden_size', type=int, default=384)
    parser.add_argument('--num_layer', type=int, default=12)
    parser.add_argument('--num_head', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(f"{args.data_dir}/{args.csv_file}")
    smiles = df[args.smiles_col].tolist()
    scattering = np.load(f"{args.data_dir}/{args.scatter_file}")
    
    # Train
    model = ScatteringGraphDIT(
        max_node=args.max_node,
        hidden_size=args.hidden_size,
        num_layer=args.num_layer,
        num_head=args.num_head,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    model.fit(X_train=smiles, y_train=scattering)
    model.save_to_local('grassy_dit_checkpoint.pt')

    # example: 
    # python -m grassy_dit.train --data_dir datasets/microsource --epochs 100
