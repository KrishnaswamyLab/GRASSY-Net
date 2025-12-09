"""
Scaffold-constrained generation via clean context.

Scaffold atoms stay clean (no noise) throughout denoising.
Self-attention lets generated atoms attend to scaffold → natural compatibility.
"""

import torch
from typing import List, Optional


def create_scaffold_mask(scaffold_atoms: List[int], num_atoms: int, device=None):
    """Create mask from scaffold atom indices."""
    node_mask = torch.zeros(num_atoms, dtype=torch.bool, device=device)
    node_mask[scaffold_atoms] = True
    # Edge mask: only edges BETWEEN scaffold atoms (internal bonds)
    edge_mask = node_mask[:, None] & node_mask[None, :]
    return {'nodes': node_mask, 'edges': edge_mask}


class ScaffoldSampler:
    """
    Sampler with CFG and scaffold constraints.
    
    Scaffold stays clean throughout — self-attention handles the rest.
    
    Usage:
        sampler = ScaffoldSampler(model)
        X, E = sampler.sample(
            scattering=target_scatter,
            num_atoms=20,
            scaffold_atoms=[0,1,2,3,4,5],
            scaffold_X=benzene_X,
            scaffold_E=benzene_E,
        )
    """
    
    def __init__(self, model, num_steps=500):
        self.model = model
        self.num_steps = num_steps
        self.device = next(model.parameters()).device
        self.model.eval()
    
    @torch.no_grad()
    def sample(
        self,
        scattering: torch.Tensor,
        num_atoms: int,
        guidance_scale: float = 2.0,
        scaffold_atoms: Optional[List[int]] = None,
        scaffold_X: Optional[torch.Tensor] = None,
        scaffold_E: Optional[torch.Tensor] = None,
    ):
        N = self.model.max_n_nodes
        Xdim = self.model.out_layer.atom_type
        Edim = self.model.out_layer.bond_type
        
        # Init from noise
        X = torch.randn(1, N, Xdim, device=self.device) * 0.1
        E = torch.randn(1, N, N, Edim, device=self.device) * 0.1
        node_mask = torch.zeros(1, N, dtype=torch.bool, device=self.device)
        node_mask[0, :num_atoms] = True
        scattering = scattering.unsqueeze(0).to(self.device)
        
        # Scaffold setup
        use_scaffold = scaffold_atoms is not None
        if use_scaffold:
            smask = create_scaffold_mask(scaffold_atoms, N, self.device)
            smask = {k: v.unsqueeze(0) for k, v in smask.items()}
            scaffold_X = scaffold_X.unsqueeze(0).to(self.device)
            scaffold_E = scaffold_E.unsqueeze(0).to(self.device)
            
            # Inject clean scaffold into initial state
            X[:, smask['nodes'][0]] = scaffold_X[:, smask['nodes'][0]]
            E[:, smask['edges'][0]] = scaffold_E[:, smask['edges'][0]]
        
        # Denoise
        for t in reversed(range(self.num_steps)):
            t_tensor = torch.tensor([t], device=self.device)
            
            # CFG
            X_c, E_c, _ = self.model(X, E, node_mask, t_tensor, scattering, uncond=False)
            if guidance_scale != 1.0:
                X_u, E_u, _ = self.model(X, E, node_mask, t_tensor, scattering, uncond=True)
                X_pred = X_u + guidance_scale * (X_c - X_u)
                E_pred = E_u + guidance_scale * (E_c - E_u)
            else:
                X_pred, E_pred = X_c, E_c

            # Optional if we want to fix scaffold siize but use pretrained model - if we remove it we need to change this classses definition as it assumes we do get a num attoms
            if num_atoms is not None:
                NO_ATOM = 0
                X_pred[:, :num_atoms, NO_ATOM] = -float('inf')   # real atoms can't be "no atom"
                X_pred[:, num_atoms:, 1:] = -float('inf')        # padding must be "no atom"

            
            # Keep scaffold clean, only update generated atoms
            if use_scaffold:
                X_pred[:, smask['nodes'][0]] = scaffold_X[:, smask['nodes'][0]]
                E_pred[:, smask['edges'][0]] = scaffold_E[:, smask['edges'][0]]
            
            X, E = X_pred, E_pred
        
        return X.squeeze(0)[:num_atoms], E.squeeze(0)[:num_atoms, :num_atoms]
