"""
GRASSY-DiT training using torch-molecule's GraphDIT infrastructure.
Replaces their property conditioning with our cross-attention to scattering tokens.
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import wandb
from torch_molecule import GraphDITMolecularGenerator
from torch_molecule.generator.graph_dit.utils import PlaceHolder
from grassy_dit.model import ScatteringDenoiser


# python -m grassy_dit.train --data_dir grassy_dit/data/moses --epochs 20 --checkpoint_dir ./checkpoints 
    
class ScatteringTransformerAdapter(torch.nn.Module):
    """Wraps ScatteringDenoiser to match Transformer.forward(noisy_data, unconditioned) signature."""
    
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser
        self.step = 0  # Track training step for Wandb logging
    
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
        flat_pred_X = masked_pred_X[mask_X, :]
        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = F.cross_entropy(flat_pred_X, torch.argmax(flat_true_X, dim=-1)) if true_X.numel() > 0 else 0.0
        loss_E = F.cross_entropy(flat_pred_E, torch.argmax(flat_true_E, dim=-1)) if true_E.numel() > 0 else 0.0
        loss = lw_X * loss_X + lw_E * loss_E
        
        # Log to Wandb
        if isinstance(loss_X, torch.Tensor):
            loss_X_val = loss_X.item()
        else:
            loss_X_val = loss_X
        if isinstance(loss_E, torch.Tensor):
            loss_E_val = loss_E.item()
        else:
            loss_E_val = loss_E
        if isinstance(loss, torch.Tensor):
            loss_val = loss.item()
        else:
            loss_val = loss
        
        wandb.log({
            "train_loss": loss_val,
            "train_loss_X": loss_X_val,
            "train_loss_E": loss_E_val,
            "step": self.step
        })
        self.step += 1
        
        return loss, loss_X, loss_E

    # just a toourch requirement, actual intialization of parameters is done in the ScatteringDenoiser class
    def initialize_parameters(self):
        """Required by torch-molecule's fit()."""
        pass


class ScatteringGraphDIT(GraphDITMolecularGenerator):
    """GraphDIT with scattering moment conditioning via cross-attention."""
    
    def __init__(self, checkpoint_dir='./checkpoints', **kwargs):
        super().__init__(**kwargs)
        self.checkpoint_dir = checkpoint_dir
        self._best_loss = float('inf')

    def _validate_inputs(self, X, y, num_task=None, num_pretask=None, return_rdkit_mol=False):
        """Compute num_atom_types from scattering dimension."""
        if y is not None:
            # Compute num_atom_types from scattering dimension
            # scattering_dim = num_atom_types * num_levels * num_moments
            num_levels = 11
            num_moments = 4
            if hasattr(y, 'shape'):
                scattering_dim = y.shape[-1] if len(y.shape) > 1 else len(y)
            else:
                # Handle list/array
                scattering_dim = len(y[0]) if len(y) > 0 else len(y)
            
            num_atom_types = scattering_dim // (num_levels * num_moments)
            self.num_atom_types = num_atom_types
            self.num_levels = num_levels
            self.num_moments = num_moments
            
            print(f"Detected scattering dimension: {scattering_dim}")
            print(f"Computed num_atom_types: {num_atom_types} (should equal in_channels from Scatter model)")
            assert scattering_dim == num_atom_types * num_levels * num_moments, \
                f"Scattering dimension {scattering_dim} must equal num_atom_types * {num_levels} * {num_moments}"
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
            num_atom_types=getattr(self, 'num_atom_types', 16),  # Use computed value or default
            num_levels=getattr(self, 'num_levels', 11),
            num_moments=getattr(self, 'num_moments', 4),
            device=self.device,
        )
        self.model = ScatteringTransformerAdapter(denoiser).to(self.device)
        
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        return self.model
    
    def _train_epoch(self, train_loader, optimizer, epoch, global_pbar=None):
        """Override to save best checkpoint."""
        loss, loss_X, loss_E = super()._train_epoch(train_loader, optimizer, epoch, global_pbar)
        
        # Check if this is the best epoch
        current_epoch = epoch + 1
    
        # Print epoch summary
        print(f"Epoch {current_epoch}/{self.epochs} - Loss: {loss:.6f} - Best: {self._best_loss:.6f}")

        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                "epoch": current_epoch,
                "epoch_loss": loss,
                "best_loss": self._best_loss,
            })
            
        if self.checkpoint_dir and loss < self._best_loss:
            self._best_loss = loss
            self._save_best_checkpoint(epoch + 1, loss)
        
        return loss, loss_X, loss_E

    def _save_best_checkpoint(self, epoch, loss):
        """Save the best checkpoint, removing previous best."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint_best.pt")
        
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "hyperparameters": {
                "max_node": self.max_node,
                "hidden_size": self.hidden_size,
                "num_layer": self.num_layer,
                "num_head": self.num_head,
                "mlp_ratio": self.mlp_ratio,
                "dropout": self.dropout,
                "drop_condition": self.drop_condition,
                "input_dim_X": self.input_dim_X,
                "input_dim_E": self.input_dim_E,
                "input_dim_y": self.input_dim_y,
                "task_type": self.task_type,
                "timesteps": self.timesteps,
                "dataset_info": self.dataset_info,
                "num_atom_types": getattr(self, 'num_atom_types', None),
                "num_levels": getattr(self, 'num_levels', None),
                "num_moments": getattr(self, 'num_moments', None),
            },
            "fitting_epoch": epoch,
            "fitting_loss": self.fitting_loss,
            "best_loss": loss,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"New best checkpoint at epoch {epoch} (loss: {loss:.6f})")
        
        if wandb.run is not None:
            wandb.save(checkpoint_path)
            wandb.log({"best_loss": loss, "best_epoch": epoch})


    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Override fit to add Wandb logging."""
        # Call parent fit which will trigger _validate_inputs and _initialize_model
        result = super().fit(X_train=X_train, y_train=y_train, **kwargs)
            
        # Log hyperparameters
        if hasattr(self, 'num_atom_types'):
            wandb.config.update({
                "num_atom_types": self.num_atom_types,
                "num_levels": self.num_levels,
                "num_moments": self.num_moments,
                "scattering_dim": self.num_atom_types * self.num_levels * self.num_moments,
            })
        
        return result
    
    @torch.no_grad()
    def generate(self, scattering, num_nodes=None, batch_size=1,
             scaffold_X=None, scaffold_E=None, scaffold_node_mask=None):
        """Generate with optional scaffold constraint."""
        
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
    parser.add_argument('--checkpoint_dir', default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_every_n_epochs', type=int, default=10, help='Save checkpoint every N epochs')
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
        entity="grassy",
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

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Train with checkpoint saving
    model = ScatteringGraphDIT(
        hidden_size=args.hidden_size,
        num_layer=args.num_layer,
        num_head=args.num_head,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir, 
    )
    
    # Compute num_atom_types from scattering data before initializing model (needed for checkpoint resume)
    num_levels = 11
    num_moments = 4
    scattering_dim = scattering.shape[-1] if len(scattering.shape) > 1 else len(scattering)
    num_atom_types = scattering_dim // (num_levels * num_moments)
    model.num_atom_types = num_atom_types
    model.num_levels = num_levels
    model.num_moments = num_moments

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
