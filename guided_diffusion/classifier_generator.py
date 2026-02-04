"""
Classifier-Guided Graph Diffusion Transformer for Molecular Generation.

Extends torch-molecule's GraphDITMolecularGenerator with classifier-based
moment guidance. Uses a trained moment classifier to compute gradients
on the noisy graph state.

Key difference from GuidedGraphDIT (in generator.py):
- GuidedGraphDIT: Computes scattering on predicted clean graph (pred_X, pred_E)
- ClassifierGuidedGraphDIT: Uses classifier on noisy graph (X_t, E_t)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, List, Tuple

from torch_molecule import GraphDITMolecularGenerator
from torch_molecule.generator.graph_dit.diffusion import (
    reverse_diffusion,
    sample_discrete_features,
)
from torch_molecule.generator.graph_dit.utils import PlaceHolder

from .classifier_guidance import (
    ClassifierGuidance,
    apply_classifier_guidance_to_probs,
)


class ClassifierGuidedGraphDIT(GraphDITMolecularGenerator):
    """
    GraphDIT with classifier-based moment guidance during generation.
    
    This class extends torch-molecule's GraphDITMolecularGenerator to support
    gradient-based guidance using a trained moment classifier. The classifier
    predicts scattering moments from noisy graphs, and its gradients are used
    to steer generation toward target moments.
    
    The guidance is applied at each reverse diffusion step by:
    1. Computing classifier gradients on current noisy state (X_t, E_t)
    2. Using these gradients to shift the model's predictions (pred_X, pred_E)
    3. Sampling the next state from the shifted distribution
    
    Example
    -------
    >>> model = ClassifierGuidedGraphDIT()
    >>> model.load_from_local("dit_checkpoint.pt")
    >>> 
    >>> # Generate with classifier guidance
    >>> smiles = model.classifier_guided_generate(
    ...     classifier_path="classifier_checkpoint.pt",
    ...     target_moments=target_moments,
    ...     guidance_scale=1.0,
    ...     num_nodes=10,
    ...     batch_size=32
    ... )
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Classifier guidance attributes (set during generation)
        self._classifier_guidance: Optional[ClassifierGuidance] = None
        self._target_moments: Optional[torch.Tensor] = None
        self._guidance_scale: float = 1.0
        self._guidance_start_step: int = 0
        self._guidance_end_step: Optional[int] = None
        self._current_step: int = 0
    
    def classifier_guided_generate(
        self,
        classifier_path: str,
        target_moments: Union[np.ndarray, torch.Tensor],
        num_nodes: Optional[Union[int, List[int], np.ndarray, torch.Tensor]] = None,
        batch_size: int = 32,
        guidance_scale: float = 1.0,
        guidance_start_step: int = 0,
        guidance_end_step: Optional[int] = None,
        labels: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
    ) -> List[str]:
        """
        Generate molecules with classifier-based moment guidance.
        
        Parameters
        ----------
        classifier_path : str
            Path to trained MomentClassifier checkpoint.
        target_moments : array-like
            Target scattering moments [D] or [B, D]. If 1D, will be expanded
            to match batch_size.
        num_nodes : int or array-like, optional
            Number of nodes per molecule. If None, samples from training distribution.
        batch_size : int, default=32
            Number of molecules to generate.
        guidance_scale : float, default=1.0
            Scale for guidance gradients. Higher = stronger guidance but may
            produce invalid molecules. Start with 1.0 and tune.
        guidance_start_step : int, default=0
            Timestep index at which to start applying guidance.
        guidance_end_step : int, optional
            Timestep index at which to stop applying guidance. If None, guidance
            continues until the end.
        labels : array-like, optional
            Property labels for conditional generation (passed to base model).
        
        Returns
        -------
        List[str]
            Generated SMILES strings.
        """
        # Convert target moments to tensor
        if isinstance(target_moments, np.ndarray):
            target_moments = torch.from_numpy(target_moments).float()
        
        # Expand to batch if needed
        if target_moments.dim() == 1:
            target_moments = target_moments.unsqueeze(0).expand(batch_size, -1).clone()
        
        target_moments = target_moments.to(self.device)
        
        # Initialize classifier guidance module
        self._classifier_guidance = ClassifierGuidance(
            classifier=classifier_path,
            device=self.device,
        )
        
        # Store guidance parameters
        self._target_moments = target_moments
        self._guidance_scale = guidance_scale
        self._guidance_start_step = guidance_start_step
        self._guidance_end_step = guidance_end_step
        self._current_step = 0
        
        # Convert num_nodes to tensor format expected by parent
        if num_nodes is not None and isinstance(num_nodes, int):
            num_nodes = torch.tensor([[num_nodes]] * batch_size, device=self.device)
        
        try:
            # Call parent generate - our overridden sample_p_zs_given_zt applies guidance
            return super().generate(
                labels=labels,
                num_nodes=num_nodes,
                batch_size=batch_size,
            )
        finally:
            # Clean up
            self._classifier_guidance = None
            self._target_moments = None
            self._guidance_scale = 1.0
            self._guidance_start_step = 0
            self._guidance_end_step = None
            self._current_step = 0
    
    def classifier_guided_generate_with_preloaded(
        self,
        classifier_guidance: ClassifierGuidance,
        target_moments: Union[np.ndarray, torch.Tensor],
        num_nodes: Optional[Union[int, List[int], np.ndarray, torch.Tensor]] = None,
        batch_size: int = 32,
        guidance_scale: float = 1.0,
        guidance_start_step: int = 0,
        guidance_end_step: Optional[int] = None,
        labels: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
    ) -> List[str]:
        """
        Generate with a pre-loaded classifier guidance module.
        
        Useful when generating multiple batches to avoid reloading the classifier.
        
        Parameters
        ----------
        classifier_guidance : ClassifierGuidance
            Pre-loaded ClassifierGuidance instance.
        target_moments : array-like
            Target scattering moments [D] or [B, D].
        ... (other parameters same as classifier_guided_generate)
        
        Returns
        -------
        List[str]
            Generated SMILES strings.
        """
        # Convert target moments to tensor
        if isinstance(target_moments, np.ndarray):
            target_moments = torch.from_numpy(target_moments).float()
        
        if target_moments.dim() == 1:
            target_moments = target_moments.unsqueeze(0).expand(batch_size, -1).clone()
        
        target_moments = target_moments.to(self.device)
        
        # Use pre-loaded guidance
        self._classifier_guidance = classifier_guidance
        self._target_moments = target_moments
        self._guidance_scale = guidance_scale
        self._guidance_start_step = guidance_start_step
        self._guidance_end_step = guidance_end_step
        self._current_step = 0
        
        if num_nodes is not None and isinstance(num_nodes, int):
            num_nodes = torch.tensor([[num_nodes]] * batch_size, device=self.device)
        
        try:
            return super().generate(
                labels=labels,
                num_nodes=num_nodes,
                batch_size=batch_size,
            )
        finally:
            # Don't delete classifier_guidance since it was passed in
            self._classifier_guidance = None
            self._target_moments = None
            self._guidance_scale = 1.0
            self._guidance_start_step = 0
            self._guidance_end_step = None
            self._current_step = 0
    
    def sample_p_zs_given_zt(
        self, s, t, X_t, E_t, properties, node_mask
    ):
        """
        Sample from p(z_s | z_t) with classifier-based moment guidance.
        
        This method overrides the parent to inject classifier guidance
        that steers generation toward target scattering moments.
        """
        bs, n, _ = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
        
        # ============== COMPUTE CLASSIFIER GUIDANCE ==============
        # Check if we're within the guidance window [start_step, end_step)
        in_guidance_window = (
            self._current_step >= self._guidance_start_step and
            (self._guidance_end_step is None or self._current_step < self._guidance_end_step)
        )
        
        classifier_grad_X = None
        classifier_grad_E = None
        
        if (self._classifier_guidance is not None and 
            self._target_moments is not None and
            in_guidance_window):
            
            # Get timestep for classifier (convert normalized t to integer)
            t_int = (t * self.timesteps).long().squeeze(-1)
            
            # Compute classifier gradients on noisy state
            with torch.enable_grad():
                classifier_grad_X, classifier_grad_E = self._classifier_guidance.compute_guidance(
                    X_t, E_t, t_int, node_mask, self._target_moments
                )
        
        self._current_step += 1
        # ===========================================================
        
        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": properties,
            "t": t,
            "node_mask": node_mask,
        }
        
        # Get model predictions
        pred = self.model(noisy_data, unconditioned=False)
        pred_X = F.softmax(pred.X, dim=-1)  # [bs, n, d0]
        pred_E = F.softmax(pred.E, dim=-1)  # [bs, n, n, d0]
        
        # ============== APPLY CLASSIFIER GUIDANCE ==============
        if classifier_grad_X is not None and classifier_grad_E is not None:
            # Apply guidance to shift predictions toward target moments
            # Note: we apply the gradient computed on (X_t, E_t) to (pred_X, pred_E)
            # This is valid because the classifier learned what moments the clean
            # molecule will have, and we want to shift the prediction toward that
            pred_X = apply_classifier_guidance_to_probs(
                pred_X, classifier_grad_X, self._guidance_scale
            )
            pred_E = apply_classifier_guidance_to_probs(
                pred_E, classifier_grad_E, self._guidance_scale
            )
        # ========================================================
        
        # Retrieve transition matrices
        device = pred_X.device
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, device)
        Qt = self.transition_model.get_Qt(beta_t, device)
        
        # Compute posterior using (possibly guided) predictions
        Xt_all = torch.cat([X_t, E_t.reshape(bs, n, -1)], dim=-1)
        predX_all = torch.cat([pred_X, pred_E.reshape(bs, n, -1)], dim=-1)
        
        unnormalized_probX_all = reverse_diffusion(
            predX_0=predX_all, X_t=Xt_all, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
        )
        
        unnormalized_prob_X = unnormalized_probX_all[:, :, :self.input_dim_X]
        unnormalized_prob_E = unnormalized_probX_all[
            :, :, self.input_dim_X:
        ].reshape(bs, n * n, -1)
        
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        
        prob_X = unnormalized_prob_X / torch.sum(
            unnormalized_prob_X, dim=-1, keepdim=True
        )
        prob_E = unnormalized_prob_E / torch.sum(
            unnormalized_prob_E, dim=-1, keepdim=True
        )
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])
        
        # Apply classifier-free guidance if configured (from parent)
        if self.guide_scale is not None and self.guide_scale != 1:
            pred_uncond = self.model(noisy_data, unconditioned=True)
            pred_X_uncond = F.softmax(pred_uncond.X, dim=-1)
            pred_E_uncond = F.softmax(pred_uncond.E, dim=-1)
            
            predX_all_uncond = torch.cat([pred_X_uncond, pred_E_uncond.reshape(bs, n, -1)], dim=-1)
            unnorm_prob_uncond = reverse_diffusion(
                predX_0=predX_all_uncond, X_t=Xt_all, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
            )
            
            uncon_prob_X = unnorm_prob_uncond[:, :, :self.input_dim_X]
            uncon_prob_E = unnorm_prob_uncond[:, :, self.input_dim_X:].reshape(bs, n * n, -1)
            
            uncon_prob_X[torch.sum(uncon_prob_X, dim=-1) == 0] = 1e-5
            uncon_prob_E[torch.sum(uncon_prob_E, dim=-1) == 0] = 1e-5
            
            uncon_prob_X = uncon_prob_X / torch.sum(uncon_prob_X, dim=-1, keepdim=True)
            uncon_prob_E = uncon_prob_E / torch.sum(uncon_prob_E, dim=-1, keepdim=True)
            uncon_prob_E = uncon_prob_E.reshape(bs, n, n, pred_E.shape[-1])
            
            prob_X = (
                uncon_prob_X
                * (prob_X / uncon_prob_X.clamp_min(1e-5)) ** self.guide_scale
            )
            prob_E = (
                uncon_prob_E
                * (prob_E / uncon_prob_E.clamp_min(1e-5)) ** self.guide_scale
            )
            prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True).clamp_min(1e-5)
            prob_E = prob_E / prob_E.sum(dim=-1, keepdim=True).clamp_min(1e-5)
        
        # Sample next state
        sampled_s = sample_discrete_features(prob_X, prob_E, node_mask=node_mask)
        
        X_s = F.one_hot(sampled_s.X, num_classes=self.input_dim_X).to(self.device).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.input_dim_E).to(self.device).float()
        
        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)
        
        out_one_hot = PlaceHolder(X=X_s, E=E_s, y=properties)
        
        return out_one_hot.mask(node_mask)


def load_classifier_guided_model(
    dit_checkpoint_path: str,
    device: str = "cuda",
) -> ClassifierGuidedGraphDIT:
    """
    Load a ClassifierGuidedGraphDIT model from checkpoint.
    
    Args:
        dit_checkpoint_path: Path to DiT model checkpoint
        device: Device to load model on
    
    Returns:
        ClassifierGuidedGraphDIT model ready for generation
    """
    model = ClassifierGuidedGraphDIT()
    model.load_from_local(dit_checkpoint_path)
    model.device = torch.device(device)
    
    return model
