"""
Gradient Ascent in Latent Space for Property Optimization
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List


class LatentOptimizer:
    def __init__(self, model, property_idx: int = 0, step_size: float = 0.1):
        self.model = model
        self.property_idx = property_idx
        self.step_size = step_size
        self.device = next(model.parameters()).device
    
    def _property_step(self, z):
        # Enable gradients
        z = z.clone().detach().requires_grad_(True)
        
        # Predict all properties (regression)
        y_full, y_pred, num_atoms_pred = self.model.predict(z)
        
        target_property = y_full[:, self.property_idx]
        
        loss = -target_property.mean()
        loss.backward(retain_graph=True)
        
        # Gradient ascent step
        if z.grad is not None:
            z_updated = z + self.step_size * z.grad
        else:
            z_updated = z
        
        return z_updated.detach()
    
    def optimize(self, z, n_steps = 20, return_decoded= True):
        """
        Perform iterative gradient ascent in latent space.
        
        Takes in
            z: Starting point in latent space [batch_size, latent_dim]
            n_steps: Number of optimization steps
            
        Returns:
            The path of latent points, decoded outputs (scattering moments of the path), and num_atoms
        """
        latent_trajectory = [z.detach().clone()]
        decoded_trajectory = None
        num_atoms_trajectory = None
        
        if return_decoded:
            with torch.no_grad():
                decoded = self.model.decode(z)
                decoded_trajectory = [decoded.detach().clone()]
                y_full, _, _ = self.model.predict(z)
                num_atoms = y_full[:, -1]  # Last property is num_atoms
                num_atoms_trajectory = [torch.round(num_atoms).detach().clone()]
        
        z_current = z.detach().clone()
        
        for step in range(n_steps):
            z_current = self._property_step(z_current)
            latent_trajectory.append(z_current.detach().clone())
            
            if return_decoded:
                with torch.no_grad():
                    decoded = self.model.decode(z_current)
                    decoded_trajectory.append(decoded.detach().clone())
                    y_full, _, _ = self.model.predict(z_current)
                    num_atoms = y_full[:, -1]  # Last property is num_atoms
                    num_atoms_trajectory.append(torch.round(num_atoms).detach().clone())
        
        return latent_trajectory, decoded_trajectory, num_atoms_trajectory
    
    def optimize_batch(self, z_batch, n_steps, return_decoded = True):
        z_current = z_batch.detach().clone()
        
        for _ in range(n_steps):
            z_current = self._property_step(z_current)
        
        final_decoded = None
        if return_decoded:
            with torch.no_grad():
                final_decoded = self.model.decode(z_current)
        
        return z_current, final_decoded


class MultiPropertyOptimizer:
    def __init__(self, model, property_weights = None, step_size = 0.1):
        self.model = model
        self.property_weights = property_weights
        self.step_size = step_size
        self.device = next(model.parameters()).device
    
    def _multi_property_step(self, z):
        """
        Take a single gradient ascent step optimizing multiple properties.
        Takes in:
            z: Current point in latent space
            
        Returns:
            Updated latent point z after gradient ascent step
        """
        z = z.clone().detach().requires_grad_(True)
        
        # Predict all properties (regression)
        property_pred, _, _ = self.model.predict(z)
        
        # Compute weighted loss
        loss = torch.zeros(1, device=z.device)
        for prop_idx, weight in self.property_weights.items():
            target_property = property_pred[:, prop_idx]
            loss = loss - weight * target_property.mean()  # Negative for maximization
        
        # Backward pass
        loss.backward(retain_graph=True)
        
        # Gradient ascent step
        if z.grad is not None:
            z_updated = z + self.step_size * z.grad
        else:
            z_updated = z
        
        return z_updated.detach()
    
    def optimize(self, z, n_steps = 20, return_decoded=True):
        """
        Here we perform iterative gradient ascent optimizing multiple properties.
        
        Takes in
            z: Starting point in latent space [batch_size, latent_dim]
            n_steps: Number of optimization steps
            
        Returns:
            The path of latent points, decoded outputs (scattering moments of the path), and num_atoms
        """
        latent_trajectory = [z.detach().clone()]
        decoded_trajectory = None
        num_atoms_trajectory = None
        
        if return_decoded:
            with torch.no_grad():
                decoded = self.model.decode(z)
                decoded_trajectory = [decoded.detach().clone()]
                # Get num_atoms from predictions
                y_full, _, _ = self.model.predict(z)
                num_atoms = y_full[:, -1]  # Last property is num_atoms
                num_atoms_trajectory = [torch.round(num_atoms).detach().clone()]
        
        z_current = z.detach().clone()
        
        for _ in range(n_steps):
            z_current = self._multi_property_step(z_current)
            latent_trajectory.append(z_current.detach().clone())
            
            if return_decoded:
                with torch.no_grad():
                    decoded = self.model.decode(z_current)
                    decoded_trajectory.append(decoded.detach().clone())
                    y_full, _, _ = self.model.predict(z_current)
                    num_atoms = y_full[:, -1]  # Last property is num_atoms
                    num_atoms_trajectory.append(torch.round(num_atoms).detach().clone())
        
        return latent_trajectory, decoded_trajectory, num_atoms_trajectory


class ConstrainedLatentOptimizer:
    
    def __init__(self, model, optimize_idx, constraint_ranges = None, step_size = 0.1, constraint_weight = 1.0):
        self.model = model
        self.optimize_idx = optimize_idx
        self.constraint_ranges = constraint_ranges or {}
        self.step_size = step_size
        self.constraint_weight = constraint_weight
        self.device = next(model.parameters()).device
    
    def _constrained_step(self, z):
        z = z.clone().detach().requires_grad_(True)
        
        property_pred, _, _ = self.model.predict(z)
        
        # Main objective: maximize target property
        loss = -property_pred[:, self.optimize_idx].mean()
        
        # Add constraint penalties
        for prop_idx, (min_val, max_val) in self.constraint_ranges.items():
            pred_val = property_pred[:, prop_idx]
            # Penalty if outside range
            below_min = F.relu(min_val - pred_val)
            above_max = F.relu(pred_val - max_val)
            constraint_violation = below_min + above_max
            loss = loss + self.constraint_weight * constraint_violation.mean()
        
        loss.backward(retain_graph=True)
        
        if z.grad is not None:
            z_updated = z + self.step_size * z.grad
        else:
            z_updated = z
        
        return z_updated.detach()
    
    def optimize(self, z, n_steps = 20, return_decoded = True):
        """
        Perform iterative gradient ascent with constraints in latent space. Here we optimize for 
        a target property while keeping other properties within specified ranges.

        Takes in
            z: Starting point in latent space [batch_size, latent_dim]
            n_steps: Number of optimization steps
            
        Returns:
            The path of latent points, decoded outputs (scattering moments of the path), and num_atoms
        """
        latent_trajectory = [z.detach().clone()]
        decoded_trajectory = None
        num_atoms_trajectory = None
        
        if return_decoded:
            with torch.no_grad():
                decoded = self.model.decode(z)
                decoded_trajectory = [decoded.detach().clone()]
                # Get num_atoms from predictions
                y_full, _, _ = self.model.predict(z)
                num_atoms = y_full[:, -1]  # Last property is num_atoms
                num_atoms_trajectory = [torch.round(num_atoms).detach().clone()]
        
        z_current = z.detach().clone()
        
        for _ in range(n_steps):
            z_current = self._constrained_step(z_current)
            latent_trajectory.append(z_current.detach().clone())
            
            if return_decoded:
                with torch.no_grad():
                    decoded = self.model.decode(z_current)
                    decoded_trajectory.append(decoded.detach().clone())
                    y_full, _, _ = self.model.predict(z_current)
                    num_atoms = y_full[:, -1]  # Last property is num_atoms
                    num_atoms_trajectory.append(torch.round(num_atoms).detach().clone())
        
        return latent_trajectory, decoded_trajectory, num_atoms_trajectory
