"""
GRASSY-DiT training using torch-molecule's GraphDIT infrastructure.
Replaces their property conditioning with our cross-attention to scattering tokens.
"""
import torch
import torch.nn.functional as F
import os
import wandb
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
        scattering = noisy_data['y_t']  # [B, 440] scattering passed as y - not propery labels as in the original GraphDIT
        
        X_pred, E_pred = self.denoiser(X_t, E_t, node_mask, t, scattering, uncond=unconditioned)
        E_pred = (E_pred + E_pred.transpose(1, 2)) / 2 # symmetrizes edge predictions - Bonds are undirected: edge (i,j) = edge (j,i)
        # Manual masking (avoids symmetry assertion)
        X_pred = X_pred * node_mask.unsqueeze(-1) # padding positions are masked out
        mask_2d = node_mask.unsqueeze(1) * node_mask.unsqueeze(2) # pad invalid edges (i,j) where i or j is padding
        E_pred = E_pred * mask_2d.unsqueeze(-1) # mask out padding edges 
        return PlaceHolder(X=X_pred, E=E_pred, y=None)
    
    def compute_loss(self, noisy_data, true_X, true_E, lw_X, lw_E, unconditioned=False):
        pred = self.forward(noisy_data, unconditioned=unconditioned)
        
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))
        masked_pred_X = torch.reshape(pred.X, (-1, pred.X.size(-1)))
        masked_pred_E = torch.reshape(pred.E, (-1, pred.E.size(-1)))
        
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)
        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :] # masked out padding positions known from ground truth. this seems to be a common practice so I use it for now, though it does seem there is somewhat of a logical failure in assuming we know the ground truth for the padding positions.
        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :] # same as above

        # cross_entropy expects: predictions [N, num_classes] as logits, targets [N] as class indices
        # argmax converts one-hot ground truth to integer labels: [1,0,0] → 0
        loss_X = F.cross_entropy(flat_pred_X, torch.argmax(flat_true_X, dim=-1)) if true_X.numel() > 0 else 0.0
        loss_E = F.cross_entropy(flat_pred_E, torch.argmax(flat_true_E, dim=-1)) if true_E.numel() > 0 else 0.0
        loss = lw_X * loss_X + lw_E * loss_E
        return loss, loss_X, loss_E # returning 3 of them for now, might be redundant though

    # just a toourch requirement, actual intialization of parameters is done in the ScatteringDenoiser class
    def initialize_parameters(self):
        """Required by torch-molecule's fit()."""
        pass


class ScatteringGraphDIT(GraphDITMolecularGenerator):
    """GraphDIT with scattering moment conditioning via cross-attention."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # currently skipping validation as the origina graphDiT validation is not compatible with the scattering data. might wanna add some validation later on.
    def _validate_inputs(self, X, y, num_task=None, num_pretask=None, return_rdkit_mol=False):
        """Bypass validation for 440-D scattering - just return as-is."""
        return X, y
    
    def _initialize_model(self, model_class, checkpoint=None):
        """Override to use ScatteringDenoiser instead of Transformer."""
        if checkpoint is not None:
            self._setup_diffusion_params(checkpoint)
        
        denoiser = ScatteringDenoiser(
            max_n_nodes=self.max_node,
            hidden_size=self.hidden_size,
            depth=self.num_layer,
            num_heads=self.num_head,
            Xdim=self.input_dim_X,
            Edim=self.input_dim_E,
        )
        self.model = ScatteringTransformerAdapter(denoiser).to(self.device)
        
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
    
        return self.model
    
    @torch.no_grad()
    def generate(self, scattering, num_nodes=None, batch_size=1,
             scaffold_X=None, scaffold_E=None, scaffold_node_mask=None):
        """Generate with optional scaffold constraint."""
        import numpy as np
        
        if isinstance(scattering, np.ndarray):
            scattering = torch.from_numpy(scattering).float()
        if scattering.dim() == 1:
            scattering = scattering.unsqueeze(0).expand(batch_size, -1).clone()
        if isinstance(num_nodes, int):
            num_nodes = torch.full((len(scattering),), num_nodes, dtype=torch.long)
        
        # No scaffold - use parent directly
        if scaffold_X is None:
            return super().generate(labels=scattering, num_nodes=num_nodes, batch_size=len(scattering))
        
        # Store scaffold for use in sample_p_zs_given_zt
        self._scaffold_X = scaffold_X.to(self.device)
        self._scaffold_E = scaffold_E.to(self.device)
        self._scaffold_node_mask = scaffold_node_mask.to(self.device)
        self._scaffold_edge_mask = scaffold_node_mask.unsqueeze(-1) & scaffold_node_mask.unsqueeze(-2)
        
        try:
            return super().generate(labels=scattering, num_nodes=num_nodes, batch_size=len(scattering))
        finally:
            # Clean up
            self._scaffold_X = None
            self._scaffold_E = None
            self._scaffold_node_mask = None
            self._scaffold_edge_mask = None

    # using the torch.mlecule method with just resetting the scaffold after each step
    def sample_p_zs_given_zt(self, s, t, X_t, E_t, properties, node_mask):
        """Override to inject scaffold after each step."""
        result = super().sample_p_zs_given_zt(s, t, X_t, E_t, properties, node_mask)
        
        # Inject scaffold if set
        if getattr(self, '_scaffold_X', None) is not None:
            result.X[self._scaffold_node_mask] = self._scaffold_X[self._scaffold_node_mask]
            result.E[self._scaffold_edge_mask] = self._scaffold_E[self._scaffold_edge_mask]
        
        return result


if __name__ == "__main__":
    import argparse
    import numpy as np
    import pandas as pd
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--csv_file', default='molecules.csv')
    parser.add_argument('--smiles_col', default='smiles')
    parser.add_argument('--scatter_file', default='scattering_moments.npy')
    parser.add_argument('--max_node', type=int, default=50)
    parser.add_argument('--hidden_size', type=int, default=384)
    parser.add_argument('--num_layer', type=int, default=12)
    parser.add_argument('--num_head', type=int, default=16) # notice that currently we force it to have the same number of cross attention heads as the number of self attention heads. we might wanna change that later
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint', default='grassy_dit_checkpoint.pt')
    parser.add_argument('--resume_from_checkpoint', default=None, type=str, help='Path to checkpoint file to resume from')
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(f"{args.data_dir}/{args.csv_file}")
    smiles = df[args.smiles_col].tolist()
    scattering = np.load(f"{args.data_dir}/{args.scatter_file}")

    # removing incopmatible with the tourch.molecule model molecules 
    from rdkit import Chem

    valid_smiles = []
    valid_scatter = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        has_dative = any(b.GetBondType() == Chem.BondType.DATIVE for b in mol.GetBonds()) # filtering out molecules with dative bonds as they are not supported by the tourch.molecule model
        if not has_dative:
            valid_smiles.append(smi)
            valid_scatter.append(scattering[i])
    
    smiles = valid_smiles
    scattering = np.array(valid_scatter)
    print(f"Filtered to {len(smiles)} molecules")
    
    # sanity check after filtering
    assert len(smiles) > 0, "No valid molecules after filtering"
    assert len(smiles) == len(scattering), "Mismatch after filtering"
    print("Initializing model...")

    # Load checkpoint if resuming
    checkpoint = None
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        print(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location='cpu')
        print("Checkpoint loaded successfully")

    # Initialize Wandb
    wandb.init(
        project="GRASSY-DiT",
        name=f"GraphDiT_h{args.hidden_size}_l{args.num_layer}_e{args.epochs}",
        config={
            "hidden_size": args.hidden_size,
            "num_layer": args.num_layer,
            "num_head": args.num_head,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "max_node": args.max_node,
            "resume_from_checkpoint": args.resume_from_checkpoint is not None,
        }
    )


    # Train
    model = ScatteringGraphDIT(
        hidden_size=args.hidden_size,
        num_layer=args.num_layer,
        num_head=args.num_head,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    # Load checkpoint into model if resuming
    if checkpoint is not None:
        model._initialize_model(None, checkpoint=checkpoint)

    print("Model initialized. Starting training...")
    model.fit(X_train=smiles, y_train=scattering)
    print("Training complete. Saving checkpoint...")
    checkpoint_path = os.path.abspath(args.checkpoint)
    print(f"Saving checkpoint to: {checkpoint_path}")
    try:
        model.save_to_local(checkpoint_path)
        print(f"Checkpoint saved successfully!")
        # Save to Wandb
        wandb.save(checkpoint_path)
    except Exception as e:
        print(f"ERROR saving checkpoint: {e}")
        import traceback
        traceback.print_exc()
    print("Done!")
    wandb.finish()

    # example: 
    # python -m grassy_dit.train --data_dir datasets/microsource --epochs 100
