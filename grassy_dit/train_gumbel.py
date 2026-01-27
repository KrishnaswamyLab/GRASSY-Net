"""
GRASSY-DiT training using torch-molecule's GraphDIT infrastructure.
Replaces their property conditioning with our cross-attention to scattering tokens.

Includes optional Gumbel-Softmax consistency loss for enforcing that generated
molecules match their conditioning scattering moments.
"""
import torch
import torch.nn.functional as F
import numpy as np
import os
import argparse
import pandas as pd
import wandb

from torch_molecule import GraphDITMolecularGenerator
from torch_molecule.generator.graph_dit.utils import PlaceHolder, to_dense
from torch_molecule.generator.graph_dit.diffusion import reverse_diffusion, sample_discrete_feature_noise
from torch_geometric.loader import DataLoader

from rdkit import Chem

from grassy_dit.model import ScatteringDenoiser
from grassy_dit.soft_scattering import DenseSoftScattering, GumbelSoftmaxSampler
from utils.config_utils import load_config, apply_overrides
import yaml
import datetime

from typing import Dict, Any, Optional
# Usage:
#     python -m grassy_dit.train_gumbel --config configs/ZINC/BBAB/BBAB_dit_gumbel_config.yaml
#     python -m grassy_dit.train --config grassy_dit_config.yaml --override training.epochs=50
#     python -m grassy_dit.train --data_dir grassy_dit/data/moses --epochs 20 
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
        
        if wandb.run is not None:
            wandb.log({
                "train_loss": loss_val,
                "train_loss_X": loss_X_val,
                "train_loss_E": loss_E_val,
                "step": self.step
            })
        self.step += 1
        
        return loss, loss_X, loss_E

    # just a torch requirement, actual intialization of parameters is done in the ScatteringDenoiser class
    def initialize_parameters(self):
        """Required by torch-molecule's fit()."""
        pass


class DifferentiableGenerator:
    """
    Differentiable molecule generation using Gumbel-Softmax.
    
    Enables gradient flow through the full generation process for
    computing consistency loss between conditioning and generated scattering.
    """
    
    def __init__(
        self,
        model: ScatteringTransformerAdapter,
        noise_schedule,
        transition_model,
        limit_dist,
        input_dim_X: int,
        input_dim_E: int,
        max_node: int,
        timesteps: int,
        num_atom_types: int,
        active_index: Optional[torch.Tensor] = None,
        J: int = 4,
        num_moments: int = 4,
        temperature: float = 1.0,
        temperature_min: float = 0.1,
        num_generation_steps: int = 50,
        guide_scale: float = 2.0,
        device: torch.device = None,
    ):
        """
        Args:
            model: The ScatteringTransformerAdapter model
            noise_schedule: NoiseScheduleDiscrete instance
            transition_model: MarginalTransition instance
            limit_dist: Limit distribution for sampling initial noise
            input_dim_X: Number of atom type classes (active in dataset)
            input_dim_E: Number of edge type classes
            max_node: Maximum number of nodes
            timesteps: Total diffusion timesteps
            num_atom_types: Number of atom types for scattering (full, e.g. 10)
            active_index: Tensor mapping active atom indices to full 118 atom types
            J: Number of wavelet scales for scattering
            num_moments: Number of statistical moments
            temperature: Initial Gumbel-Softmax temperature
            temperature_min: Minimum temperature (annealed to during generation)
            num_generation_steps: Number of steps for generation (can be < timesteps)
            guide_scale: Classifier-free guidance scale
            device: Device for computation
        """
        self.model = model
        self.noise_schedule = noise_schedule
        self.transition_model = transition_model
        self.limit_dist = limit_dist
        self.input_dim_X = input_dim_X
        self.input_dim_E = input_dim_E
        self.max_node = max_node
        self.timesteps = timesteps
        self.num_atom_types = num_atom_types
        self.active_index = active_index
        self.temperature = temperature
        self.temperature_min = temperature_min
        self.num_generation_steps = num_generation_steps
        self.guide_scale = guide_scale
        self.device = device
        
        # Initialize soft scattering transform
        self.soft_scattering = DenseSoftScattering(
            num_atom_types=num_atom_types,
            J=J,
            num_moments=num_moments
        ).to(device)
        
    def _get_temperature(self, step, total_steps):
        """Anneal temperature from high to low during generation."""
        progress = step / max(total_steps - 1, 1)
        return self.temperature * (1 - progress) + self.temperature_min * progress
    
    def _soft_reverse_step(self, X_t, E_t, scattering, node_mask, s_norm, t_norm, temperature):
        """
        Perform one reverse diffusion step with Gumbel-Softmax sampling.
        
        Returns soft (differentiable) samples instead of hard discrete samples.
        """
        bs, n, _ = X_t.shape
        device = X_t.device
        
        beta_t = self.noise_schedule(t_normalized=t_norm)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_norm)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_norm)
        
        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": scattering,
            "t": t_norm * self.timesteps,
            "node_mask": node_mask,
        }
        
        # Get predictions (conditioned)
        pred = self.model(noisy_data, unconditioned=False)
        pred_X = F.softmax(pred.X, dim=-1)
        pred_E = F.softmax(pred.E, dim=-1)
        
        # Classifier-free guidance
        if self.guide_scale is not None and self.guide_scale != 1:
            pred_uncond = self.model(noisy_data, unconditioned=True)
            pred_X_uncond = F.softmax(pred_uncond.X, dim=-1)
            pred_E_uncond = F.softmax(pred_uncond.E, dim=-1)
            
            # Apply guidance in probability space
            pred_X = pred_X_uncond * (pred_X / pred_X_uncond.clamp_min(1e-5)) ** self.guide_scale
            pred_E = pred_E_uncond * (pred_E / pred_E_uncond.clamp_min(1e-5)) ** self.guide_scale
            pred_X = pred_X / pred_X.sum(dim=-1, keepdim=True).clamp_min(1e-5)
            pred_E = pred_E / pred_E.sum(dim=-1, keepdim=True).clamp_min(1e-5)
        
        # Get transition matrices
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device)
        Qt = self.transition_model.get_Qt(beta_t, device)
        
        # Compute reverse diffusion probabilities
        Xt_all = torch.cat([X_t, E_t.reshape(bs, n, -1)], dim=-1)
        predX_all = torch.cat([pred_X, pred_E.reshape(bs, n, -1)], dim=-1)
        
        unnormalized_probX_all = reverse_diffusion(
            predX_0=predX_all, X_t=Xt_all, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
        )
        
        unnormalized_prob_X = unnormalized_probX_all[:, :, :self.input_dim_X]
        unnormalized_prob_E = unnormalized_probX_all[:, :, self.input_dim_X:].reshape(bs, n * n, -1)
        
        # Normalize
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, self.input_dim_E)
        
        # Gumbel-Softmax sampling (differentiable)
        eps = 1e-10
        logits_X = torch.log(prob_X.clamp(min=eps))
        logits_E = torch.log(prob_E.clamp(min=eps))
        
        X_s = F.gumbel_softmax(logits_X, tau=temperature, hard=False, dim=-1)
        
        # Sample edges
        logits_E_flat = logits_E.reshape(-1, self.input_dim_E)
        E_s_flat = F.gumbel_softmax(logits_E_flat, tau=temperature, hard=False, dim=-1)
        E_s = E_s_flat.reshape(bs, n, n, self.input_dim_E)
        
        # Make edges symmetric
        E_s = (E_s + E_s.transpose(1, 2)) / 2
        
        # Mask invalid nodes/edges
        node_mask_expanded = node_mask.unsqueeze(-1).float()
        edge_mask = (node_mask.unsqueeze(1) & node_mask.unsqueeze(2)).unsqueeze(-1).float()
        
        X_s = X_s * node_mask_expanded
        E_s = E_s * edge_mask
        
        return X_s, E_s
    
    def generate_soft(self, scattering, node_mask):
        """
        Generate molecules with differentiable Gumbel-Softmax sampling.
        
        Args:
            scattering: [B, scattering_dim] conditioning scattering moments
            node_mask: [B, N] valid nodes mask
            
        Returns:
            soft_X: [B, N, input_dim_X] soft atom type probabilities
            soft_E: [B, N, N, input_dim_E] soft edge type probabilities
        """
        bs = scattering.shape[0]
        device = scattering.device
        
        # Sample initial noise
        z_T = sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        X, E = z_T.X.to(device), z_T.E.to(device)
        
        # Determine step size for generation
        step_size = max(1, self.timesteps // self.num_generation_steps)
        steps = list(range(0, self.timesteps, step_size))
        if steps[-1] != self.timesteps - 1:
            steps.append(self.timesteps - 1)
        steps = list(reversed(steps))
        
        # Reverse diffusion with Gumbel-Softmax
        for i, s_int in enumerate(steps[1:]):
            t_int = steps[i]
            
            s_array = torch.full((bs, 1), s_int, dtype=torch.float, device=device)
            t_array = torch.full((bs, 1), t_int, dtype=torch.float, device=device)
            s_norm = s_array / self.timesteps
            t_norm = t_array / self.timesteps
            
            # Anneal temperature
            temperature = self._get_temperature(i, len(steps) - 1)
            
            # Soft reverse step
            X, E = self._soft_reverse_step(X, E, scattering, node_mask, s_norm, t_norm, temperature)
        
        return X, E
    
    def _expand_to_full_atom_types(self, soft_X):
        """
        Expand soft_X from active atom types to full num_atom_types.
        
        The model operates on input_dim_X active atom types, but scattering
        was computed with num_atom_types (e.g., 10). We need to map back.
        
        Args:
            soft_X: [B, N, input_dim_X] soft atom predictions (active types only)
            
        Returns:
            expanded_X: [B, N, num_atom_types] with zeros for inactive types
        """
        if self.input_dim_X == self.num_atom_types:
            return soft_X  # No expansion needed
        
        B, N, _ = soft_X.shape
        
        # Create full tensor with zeros
        expanded_X = torch.zeros(B, N, self.num_atom_types, device=soft_X.device, dtype=soft_X.dtype)
        
        # Map active indices to full tensor
        # active_index maps: position i in input_dim_X -> atom type active_index[i] in full 118
        # But for scattering, we just need the first num_atom_types
        # The active_index contains the actual atom numbers (e.g., [6, 7, 8] for C, N, O)
        # We need to map these to positions 0, 1, 2, ... in num_atom_types
        
        if self.active_index is not None:
            # active_index is like [6, 7, 8, 9, 16, ...] (atomic numbers - 1)
            # For scattering with 10 atom types, we assume the first num_atom_types 
            # elements of active_index correspond to the scattering atom types
            for i in range(min(self.input_dim_X, self.num_atom_types)):
                expanded_X[:, :, i] = soft_X[:, :, i]
        else:
            # No active_index - just copy directly
            expanded_X[:, :, :self.input_dim_X] = soft_X
        
        return expanded_X
    
    def compute_consistency_loss(self, scattering_cond, node_mask):
        """
        Compute consistency loss between conditioning and generated scattering.
        
        Args:
            scattering_cond: [B, scattering_dim] conditioning scattering moments
            node_mask: [B, N] valid nodes mask
            
        Returns:
            loss: Scalar L2 loss between conditioning and generated scattering
            scattering_gen: [B, scattering_dim] generated scattering (for logging)
        """
        # Generate soft molecules
        soft_X, soft_E = self.generate_soft(scattering_cond, node_mask)
        
        # Expand soft_X to full atom type dimensions for scattering
        soft_X_expanded = self._expand_to_full_atom_types(soft_X)
        
        # Compute scattering of generated molecules
        scattering_gen = self.soft_scattering(soft_X_expanded, soft_E, node_mask)
        
        # L2 loss
        loss = F.mse_loss(scattering_gen, scattering_cond)
        
        return loss, scattering_gen


class ScatteringGraphDIT(GraphDITMolecularGenerator):
    """GraphDIT with scattering moment conditioning via cross-attention.
    
    Supports optional Gumbel-Softmax consistency loss for enforcing that
    generated molecules match their conditioning scattering moments.
    """
    
    def __init__(self, config: Dict[str, Any], **kwargs):
        model_cfg = config.get('model', {})
        training_cfg = config.get('training', {})
        checkpoint_cfg = config.get('checkpoint', {})
        consistency_cfg = config.get('consistency', {})
        
        super().__init__(
            hidden_size=model_cfg.get('hidden_size', 384),
            num_layer=model_cfg.get('num_layer', 12),
            num_head=model_cfg.get('num_head', 16),
            # mlp_ratio=model_cfg.get('mlp_ratio', 4.0),
            # dropout=model_cfg.get('dropout', 0.1),
            # drop_condition=model_cfg.get('drop_condition', 0.1),
            epochs=training_cfg.get('epochs', 100),
            batch_size=training_cfg.get('batch_size', 32),
            learning_rate=training_cfg.get('learning_rate', 1e-4),
            **kwargs
        )
        
        self.config = config
        self.checkpoint_dir = checkpoint_cfg.get('save_dir', './checkpoints')
        self.save_every_n_epochs = checkpoint_cfg.get('save_every_n_epochs', 10)
        self._best_loss = float('inf')
        
        # Consistency loss configuration
        self.use_consistency_loss = consistency_cfg.get('enabled', False)
        self.lw_consistency = consistency_cfg.get('weight', 0.1)
        self.consistency_temperature = consistency_cfg.get('temperature', 1.0)
        self.consistency_temperature_min = consistency_cfg.get('temperature_min', 0.1)
        self.consistency_num_steps = consistency_cfg.get('num_generation_steps', 50)
        self.consistency_every_n_steps = consistency_cfg.get('every_n_steps', 10)
        self.consistency_start_epoch = consistency_cfg.get('start_epoch', 5)
        
        # Will be initialized after model is created
        self._diff_generator = None

    def _validate_inputs(self, X, y, num_task=None, num_pretask=None, return_rdkit_mol=False):
        """Compute num_atom_types from scattering dimension."""
        if y is not None:
            scattering_cfg = self.config.get('scattering', {})
            J = scattering_cfg.get('J', 4)
            num_levels = 1 + J + J*(J-1)//2 
            num_moments = scattering_cfg.get('num_moments', 4)
            
            if hasattr(y, 'shape'):
                scattering_dim = y.shape[-1] if len(y.shape) > 1 else len(y)
            else:
                scattering_dim = len(y[0]) if len(y) > 0 else len(y)
            
            num_atom_types = scattering_dim // (num_levels * num_moments)
            self.num_atom_types = num_atom_types
            self.num_levels = num_levels
            self.num_moments = num_moments

            print(f"Detected scattering dimension: {scattering_dim}")
            print(f"Computed num_atom_types: {num_atom_types}")
            
            assert scattering_dim == num_atom_types * num_levels * num_moments, \
                f"Scattering dimension {scattering_dim} must equal num_atom_types * {num_levels} * {num_moments}"
        return X, y
    
    
    def _initialize_model(self, model_class, checkpoint=None):
        """Override to use ScatteringDenoiser instead of Transformer."""
        if checkpoint is not None:
            self._setup_diffusion_params(checkpoint)
        
        model_cfg = self.config.get('model', {})
        scattering_cfg = self.config.get('scattering', {})
        
        denoiser = ScatteringDenoiser(
            max_n_nodes=self.max_node,
            hidden_size=self.hidden_size,
            depth=self.num_layer,
            num_heads=self.num_head,
            Xdim=self.input_dim_X,
            Edim=self.input_dim_E,
            num_atom_types=getattr(self, 'num_atom_types', 16),
            num_levels=getattr(self, 'num_levels', 11),
            num_moments=getattr(self, 'num_moments', 4),
            device=self.device,
        )
        self.model = ScatteringTransformerAdapter(
            denoiser
        ).to(self.device)
        
        if checkpoint is not None:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        
        # Initialize differentiable generator for consistency loss
        if self.use_consistency_loss:
            # Get active_index if available (maps model atom types to full atom types)
            active_index = None
            if hasattr(self, 'dataset_info') and self.dataset_info:
                active_index = self.dataset_info.get("active_index", None)
            
            self._diff_generator = DifferentiableGenerator(
                model=self.model,
                noise_schedule=self.noise_schedule,
                transition_model=self.transition_model,
                limit_dist=self.limit_dist,
                input_dim_X=self.input_dim_X,
                input_dim_E=self.input_dim_E,
                max_node=self.max_node,
                timesteps=self.timesteps,
                num_atom_types=getattr(self, 'num_atom_types', 16),
                active_index=active_index,
                J=scattering_cfg.get('J', 4),
                num_moments=scattering_cfg.get('num_moments', 4),
                temperature=self.consistency_temperature,
                temperature_min=self.consistency_temperature_min,
                num_generation_steps=self.consistency_num_steps,
                guide_scale=self.guide_scale,
                device=self.device,
            )
            print(f"Initialized DifferentiableGenerator for consistency loss")
            print(f"  - Weight: {self.lw_consistency}")
            print(f"  - Temperature: {self.consistency_temperature} -> {self.consistency_temperature_min}")
            print(f"  - Generation steps: {self.consistency_num_steps}")
            print(f"  - Every N training steps: {self.consistency_every_n_steps}")
            print(f"  - Start epoch: {self.consistency_start_epoch}")
        
        return self.model
    
    def _train_epoch(self, train_loader, optimizer, epoch, global_pbar=None):
        """Override to add consistency loss, validation, and checkpoint saving."""
        print(f"Starting epoch {epoch}...", flush=True)
        
        # Check if we should use consistency loss this epoch
        current_epoch = epoch + 1
        use_consistency = (
            self.use_consistency_loss and 
            self._diff_generator is not None and
            current_epoch >= self.consistency_start_epoch
        )
        
        if use_consistency:
            # Custom training loop with consistency loss
            losses = self._train_epoch_with_consistency(train_loader, optimizer, epoch, global_pbar)
        else:
            # Standard training
            losses = super()._train_epoch(train_loader, optimizer, epoch, global_pbar)
        
        avg_train_loss = sum(losses) / len(losses)
        
        # Save checkpoint FIRST if train loss improved
        if self.checkpoint_dir and avg_train_loss < self._best_loss:
            self._best_loss = avg_train_loss
            self._save_best_checkpoint(current_epoch, avg_train_loss)
        
        # Run validation every N epochs (if val data exists)
        val_loss = None
        val_every = getattr(self, '_val_every_n_epochs', 1)
        if getattr(self, '_val_smiles', None) is not None and current_epoch % val_every == 0:
            val_loss = self._compute_val_loss()
            if val_loss is not None and self.checkpoint_dir and val_loss < self._best_loss:
                self._best_loss = val_loss
                self._save_best_checkpoint(current_epoch, val_loss)
        
        # Logging
        loss_str = f"Epoch {current_epoch}/{self.epochs} - Train: {avg_train_loss:.6f}"
        if val_loss is not None:
            loss_str += f" - Val: {val_loss:.6f}"
        loss_str += f" - Best: {self._best_loss:.6f}"
        print(loss_str)
        
        if wandb.run is not None:
            log_dict = {"epoch": current_epoch, "train_loss_epoch": avg_train_loss, "best_loss": self._best_loss}
            if val_loss is not None:
                log_dict["val_loss"] = val_loss
            wandb.log(log_dict)
        
        return losses
    
    def _train_epoch_with_consistency(self, train_loader, optimizer, epoch, global_pbar=None):
        """Training epoch with Gumbel-Softmax consistency loss."""
        self.model.train()
        losses = []
        active_index = self.dataset_info["active_index"]
        
        global_step = epoch * len(train_loader)
        
        for step, batched_data in enumerate(train_loader):
            batched_data = batched_data.to(self.device)
            optimizer.zero_grad()
            
            # Convert to dense format
            data_x = F.one_hot(batched_data.x, num_classes=118).float()[:, active_index]
            data_edge_attr = F.one_hot(batched_data.edge_attr, num_classes=5).float()
            dense_data, node_mask = to_dense(data_x, batched_data.edge_index, data_edge_attr, batched_data.batch, self.max_node)
            dense_data = dense_data.mask(node_mask)
            X, E = dense_data.X, dense_data.E
            
            # Standard denoising loss
            noisy_data = self.apply_noise(X, E, batched_data.y, node_mask)
            loss_denoise, loss_X, loss_E = self.model.compute_loss(
                noisy_data, true_X=X, true_E=E, lw_X=self.lw_X, lw_E=self.lw_E
            )
            
            # Consistency loss (every N steps to save compute)
            loss_consistency = torch.tensor(0.0, device=self.device)
            if (global_step + step) % self.consistency_every_n_steps == 0:
                scattering_cond = batched_data.y.to(self.device)
                loss_consistency, _ = self._diff_generator.compute_consistency_loss(
                    scattering_cond, node_mask
                )
            
            # Total loss
            loss = loss_denoise + self.lw_consistency * loss_consistency
            
            loss.backward()
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)
            optimizer.step()
            
            losses.append(loss.item())
            
            # Logging
            log_dict = {
                "Epoch": f"{epoch+1}/{self.epochs}",
                "Step": f"{step+1}/{len(train_loader)}",
                "Loss": f"{loss.item():.4f}",
                "Loss_X": f"{loss_X.item() if isinstance(loss_X, torch.Tensor) else loss_X:.4f}",
                "Loss_E": f"{loss_E.item() if isinstance(loss_E, torch.Tensor) else loss_E:.4f}",
                "Loss_Consistency": f"{loss_consistency.item():.4f}",
            }
            
            if global_pbar is not None:
                global_pbar.set_postfix(log_dict)
                global_pbar.update(1)
            
            if wandb.run is not None:
                wandb.log({
                    "train_loss": loss.item(),
                    "train_loss_X": loss_X.item() if isinstance(loss_X, torch.Tensor) else loss_X,
                    "train_loss_E": loss_E.item() if isinstance(loss_E, torch.Tensor) else loss_E,
                    "train_loss_consistency": loss_consistency.item(),
                    "step": global_step + step,
                })
        
        return losses

    @torch.no_grad()
    def _compute_val_loss(self):
        """Compute validation loss using parent's data processing methods."""
        if self._val_smiles is None or len(self._val_smiles) == 0:
            return None
        
        val_dataset = self._convert_to_pytorch_data(self._val_smiles, self._val_scattering)
        if len(val_dataset) == 0:
            print("Warning: No valid validation samples after conversion")
            return None
            
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        active_index = self.dataset_info["active_index"]
        
        for batched_data in val_loader:
            batched_data = batched_data.to(self.device)
            data_x = F.one_hot(batched_data.x, num_classes=118).float()[:, active_index]
            data_edge_attr = F.one_hot(batched_data.edge_attr, num_classes=5).float()
            dense_data, node_mask = to_dense(data_x, batched_data.edge_index, data_edge_attr, batched_data.batch, self.max_node)
            dense_data = dense_data.mask(node_mask)
            X, E = dense_data.X, dense_data.E
            noisy_data = self.apply_noise(X, E, batched_data.y, node_mask)
            loss, _, _ = self.model.compute_loss(noisy_data, true_X=X, true_E=E, lw_X=self.lw_X, lw_E=self.lw_E)
            total_loss += loss.item()
            num_batches += 1
        
        self.model.train()
        return total_loss / num_batches if num_batches > 0 else None

    def _save_best_checkpoint(self, epoch, loss):
        """Save the best checkpoint, removing previous best."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_best.pt")
        
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


    def fit(self, X_train, y_train, X_val=None, y_val=None, val_every_n_epochs=1, **kwargs):
        """Override fit to add validation and Wandb logging."""
        # Store validation data for use in _train_epoch
        self._val_smiles = X_val
        self._val_scattering = y_val
        self._val_every_n_epochs = val_every_n_epochs
        
        # Call parent fit which will trigger _validate_inputs and _initialize_model
        result = super().fit(X_train=X_train, y_train=y_train, **kwargs)
            
        # Log hyperparameters
        if hasattr(self, 'num_atom_types') and wandb.run is not None:
            wandb.config.update({
                "num_atom_types": self.num_atom_types,
                "num_levels": self.num_levels,
                "num_moments": self.num_moments,
                "scattering_dim": self.num_atom_types * self.num_levels * self.num_moments,
                "val_size": len(X_val) if X_val is not None else 0,
            })
        
        return result
    
    @torch.no_grad()
    def generate(self, scattering, num_nodes=None, batch_size=1,
             scaffold_X=None, scaffold_E=None, scaffold_node_mask=None):
        """Generate with optional scaffold constraint."""
        self._is_fitted = True  # Required for torch_molecule's generate() check
        
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

    parser = argparse.ArgumentParser(
        description='Train GRASSY-DiT with scattering conditioning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            # Use config file
            python -m grassy_dit.train --config grassy_dit_config.yaml
            
            # Override specific values
            python -m grassy_dit.train --config grassy_dit_config.yaml --override training.epochs=50 model.hidden_size=512
            
            # Legacy CLI mode (without config file)
            python -m grassy_dit.train --data_dir grassy_dit/data/moses --epochs 20
        """
    )
    
    # Config-based arguments
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    parser.add_argument('--override', type=str, nargs='*', default=[],
                        help='Override config values (e.g., training.epochs=50)')
    
    args = parser.parse_args()
    
    # =========================================================================
    # Load and process config
    # =========================================================================
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)
    
    if args.override:
        print("Applying overrides:")
        config = apply_overrides(config, args.override)

    # Extract config sections
    data_cfg = config['dataset']
    model_cfg = config['model']
    training_cfg = config['training']
    checkpoint_cfg = config['checkpoint']
    wandb_cfg = config.get('wandb', {})

    # =========================================================================
    # Create checkpoint directory and save config
    # =========================================================================
    checkpoint_dir = config['checkpoint']['save_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    now = datetime.datetime.now()
    date_suffix = now.strftime("%Y-%m-%d-%H-%M-%S")
    config_save_path = os.path.join(checkpoint_dir, f'config_{date_suffix}.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"\nConfig saved to: {config_save_path}")

    # =========================================================================
    # Load data
    # =========================================================================
    data_dir = data_cfg['data_dir']
    csv_file = data_cfg['csv_file']
    scatter_file = data_cfg['scatter_file']
    smiles_col = data_cfg['smiles_col']
    
    df = pd.read_csv(os.path.join(data_dir, csv_file))
    smiles = df[smiles_col].tolist()
    scattering = np.load(os.path.join(data_dir, scatter_file))
    
    df = pd.read_csv(f"{data_dir}/{csv_file}")
    smiles = df[smiles_col].tolist()
    scattering = np.load(f"{data_dir}/{scatter_file}")

    # Filter incompatible molecules (dative bonds not supported by torch.molecule)
    valid_smiles = []
    valid_scatter = []
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        has_dative = any(b.GetBondType() == Chem.BondType.DATIVE for b in mol.GetBonds())
        if not has_dative:
            valid_smiles.append(smi)
            valid_scatter.append(scattering[i])
    
    smiles = valid_smiles
    scattering = np.array(valid_scatter)
    print(f"Filtered to {len(smiles)} molecules")
    
    # sanity check after filtering
    assert len(smiles) > 0, "No valid molecules after filtering"
    assert len(smiles) == len(scattering), "Mismatch after filtering"

    # =========================================================================
    # Load validation data (optional)
    # =========================================================================
    val_smiles, val_scattering = None, None
    val_data_dir = data_cfg.get('val_data_dir')
    val_every_n_epochs = data_cfg.get('val_every_n_epochs', 1)
    
    if val_data_dir:
        val_csv = data_cfg.get('val_csv_file') or csv_file
        val_scatter = data_cfg.get('val_scatter_file') or scatter_file
        
        print(f"Loading validation data from: {val_data_dir}")
        df_val = pd.read_csv(os.path.join(val_data_dir, val_csv))
        val_smiles_raw = df_val[smiles_col].tolist()
        val_scattering_raw = np.load(os.path.join(val_data_dir, val_scatter))
        
        # Filter validation molecules (same as training)
        valid_val_smiles, valid_val_scatter = [], []
        for i, smi in enumerate(val_smiles_raw):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            has_dative = any(b.GetBondType() == Chem.BondType.DATIVE for b in mol.GetBonds())
            if not has_dative:
                valid_val_smiles.append(smi)
                valid_val_scatter.append(val_scattering_raw[i])
        
        val_smiles = valid_val_smiles
        val_scattering = np.array(valid_val_scatter)
        print(f"Validation: {len(val_smiles)} molecules after filtering")

    # =========================================================================
    # Initialize model
    # =========================================================================
    print("Initializing model...")

    # Load checkpoint if resuming
    checkpoint = None
    resume_path = checkpoint_cfg.get('resume_from')
    if resume_path and os.path.exists(resume_path):
        print(f"Loading checkpoint from {resume_path}")
        checkpoint = torch.load(resume_path, map_location='cpu')
        print("Checkpoint loaded successfully")

    # Initialize Wandb
    if wandb_cfg.get('enabled', True):
        wandb.init(
            project=wandb_cfg.get('project', 'GRASSY-DiT'),
            entity=wandb_cfg.get('entity', 'grassy'),
            name=f"GraphDiT_h{model_cfg['hidden_size']}_l{model_cfg['num_layer']}_e{training_cfg['epochs']}",
            config=config,
        )

    # Create model
    model = ScatteringGraphDIT(config=config)
    
    # Compute num_atom_types from scattering data (needed for checkpoint resume)
    num_levels = model_cfg.get('num_levels', 11)
    num_moments = model_cfg.get('num_moments', 4)
    scattering_dim = scattering.shape[-1] if len(scattering.shape) > 1 else len(scattering)
    num_atom_types = scattering_dim // (num_levels * num_moments)
    model.num_atom_types = num_atom_types
    model.num_levels = num_levels
    model.num_moments = num_moments

   # Load checkpoint into model if resuming
    if checkpoint is not None:
        model._initialize_model(None, checkpoint=checkpoint)

    # =========================================================================
    # Train
    # =========================================================================
    print("Model initialized. Starting training...")
    model.fit(X_train=smiles, y_train=scattering, X_val=val_smiles, y_val=val_scattering, val_every_n_epochs=val_every_n_epochs)


    # =========================================================================
    # Save final checkpoint
    # =========================================================================
    print("Training complete. Saving checkpoint...")
    final_checkpoint_path = os.path.join(checkpoint_dir, f'final_model_{date_suffix}.pt')
    print(f"Saving checkpoint to: {final_checkpoint_path}")
    
    try:
        model.save_to_local(final_checkpoint_path)
        print("Checkpoint saved successfully!")
        
        # Save to Wandb
        if wandb_cfg.get('enabled', True):
            wandb.save(final_checkpoint_path)
    except Exception as e:
        print(f"ERROR saving checkpoint: {e}")
        import traceback
        traceback.print_exc()
    
    print("Done!")
    
    if wandb_cfg.get('enabled', True):
        wandb.finish()
