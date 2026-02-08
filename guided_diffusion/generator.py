"""
Guided Graph Diffusion Transformer for Molecular Generation.

Extends torch-molecule's GraphDITMolecularGenerator with scattering moment guidance
and optional scaffold constraints.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, List, Tuple

from rdkit import Chem

from torch_molecule import GraphDITMolecularGenerator
from torch_molecule.generator.graph_dit.diffusion import (
    reverse_diffusion,
    sample_discrete_features
)
from torch_molecule.generator.graph_dit.utils import PlaceHolder

from .guidance import ScatteringMomentGuidance, apply_guidance_to_probs


def smiles_to_scaffold(
    full_smiles: str,
    max_nodes: int,
    atom_decoder: List[str],
    bond_decoder: Optional[List[str]] = None,
    scaffold_pattern: Optional[str] = None,
    remove_indices: Optional[List[int]] = None,
    num_nodes: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Build graph from full molecule with scaffold mask indicating fixed atoms.
    
    Args:
        full_smiles: Complete molecule SMILES
        max_nodes: Maximum nodes in graph
        atom_decoder: List of atom symbols (e.g., ['C', 'N', 'O', ...])
        bond_decoder: List of bond type names (optional)
        scaffold_pattern: SMARTS pattern to KEEP - atoms matching this are fixed
        remove_indices: List of atom indices to REMOVE - atoms NOT in this list are fixed
        num_nodes: Desired output size (defaults to molecule's atom count)
        
    Exactly one of scaffold_pattern or remove_indices must be provided.
    
    Returns:
        X: [max_nodes, num_atom_types] atom features (one-hot)
        E: [max_nodes, max_nodes, num_bond_types] bond features (one-hot)
        scaffold_mask: [max_nodes] bool - True = fixed scaffold atom
        node_mask: [max_nodes] bool - True = real atom, False = padding
        n_atoms: number of atoms in full molecule
    """
    if (scaffold_pattern is None) == (remove_indices is None):
        raise ValueError("Exactly one of scaffold_pattern or remove_indices must be provided")
    
    full_mol = Chem.MolFromSmiles(full_smiles)
    if full_mol is None:
        raise ValueError(f"Invalid full molecule SMILES: {full_smiles}")
    
    n_atoms = full_mol.GetNumAtoms()
    
    # Determine which atoms are scaffold (fixed)
    if scaffold_pattern is not None:
        # Mode 1: SMARTS pattern matching - keep atoms that match
        scaffold_mol = Chem.MolFromSmarts(scaffold_pattern)
        if scaffold_mol is None:
            raise ValueError(f"Invalid SMARTS pattern: {scaffold_pattern}")
        
        match = full_mol.GetSubstructMatch(scaffold_mol)
        if not match:
            raise ValueError(f"Scaffold pattern not found in molecule")
        scaffold_indices = set(match)
    else:
        # Mode 2: Removal - keep atoms NOT in remove list
        scaffold_indices = set(range(n_atoms)) - set(remove_indices)
    
    # Build atom features from full molecule
    X = torch.zeros(max_nodes, len(atom_decoder))
    for i, atom in enumerate(full_mol.GetAtoms()):
        symbol = atom.GetSymbol()
        if symbol in atom_decoder:
            X[i, atom_decoder.index(symbol)] = 1.0
    
    # Build bond features from full molecule
    num_bond_types = len(bond_decoder) if bond_decoder else 5
    E = torch.zeros(max_nodes, max_nodes, num_bond_types)
    E[:, :, 0] = 1.0  # default no-bond
    
    if bond_decoder:
        bond_map = {}
        for idx, name in enumerate(bond_decoder):
            if name == 'SINGLE': bond_map[Chem.BondType.SINGLE] = idx
            elif name == 'DOUBLE': bond_map[Chem.BondType.DOUBLE] = idx
            elif name == 'TRIPLE': bond_map[Chem.BondType.TRIPLE] = idx
            elif name == 'AROMATIC': bond_map[Chem.BondType.AROMATIC] = idx
    else:
        bond_map = {
            Chem.BondType.SINGLE: 1,
            Chem.BondType.DOUBLE: 2,
            Chem.BondType.TRIPLE: 3,
            Chem.BondType.AROMATIC: 4,
        }
    
    for bond in full_mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = bond_map.get(bond.GetBondType(), 0)
        E[i, j, :] = 0
        E[j, i, :] = 0
        E[i, j, bt] = 1.0
        E[j, i, bt] = 1.0
    
    # Node mask: real atoms vs padding
    effective_nodes = num_nodes if num_nodes is not None else n_atoms
    node_mask = torch.zeros(max_nodes, dtype=torch.bool)
    node_mask[:effective_nodes] = True
    
    # Scaffold mask: which atoms are fixed
    scaffold_mask = torch.zeros(max_nodes, dtype=torch.bool)
    for idx in scaffold_indices:
        scaffold_mask[idx] = True

    # Zero out non-scaffold atoms (remove them from initial graph)
    X[~scaffold_mask] = 0
    E[~scaffold_mask, :, :] = 0
    E[:, ~scaffold_mask, :] = 0
    
    return X, E, scaffold_mask, node_mask, n_atoms


class GuidedGraphDIT(GraphDITMolecularGenerator):
    """
    GraphDIT with scattering moment guidance during generation.

    This class extends torch-molecule's GraphDITMolecularGenerator to support
    gradient-based guidance that steers molecule generation toward target
    scattering moments, without requiring any retraining.

    The guidance is applied at each reverse diffusion step by:
    1. Computing scattering moments from the predicted clean graph
    2. Computing MSE loss to target moments
    3. Backpropagating to get gradients w.r.t. predictions
    4. Shifting predictions in the direction that reduces moment distance

    Example
    -------
    >>> model = GuidedGraphDIT()
    >>> model.load_from_local("checkpoint.pt")
    >>>
    >>> # Compute target moments from reference molecule
    >>> target_moments = compute_scattering_from_smiles(["CCO"], scattering_model)
    >>>
    >>> # Generate with guidance
    >>> smiles = model.guided_generate(
    ...     target_moments=target_moments,
    ...     guidance_scale=1.0,
    ...     num_nodes=10,
    ...     batch_size=32
    ... )
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Guidance-specific attributes (set during guided_generate)
        self._guidance: Optional[ScatteringMomentGuidance] = None
        self._target_moments: Optional[torch.Tensor] = None
        self._guidance_scale: float = 1.0
        self._guidance_start_step: int = 0  # Step to start applying guidance
        self._guidance_end_step: Optional[int] = None  # Step to stop guidance (None = guide until end)
        self._current_step: int = 0
        self._total_steps: int = 500  # Will be set dynamically if possible
        self._diagnostic_logging: bool = True  # Enable diagnostic logging for debugging

        # Scaffold-specific attributes (set during guided_generate_with_scaffold)
        self._scaffold_X: Optional[torch.Tensor] = None
        self._scaffold_E: Optional[torch.Tensor] = None
        self._scaffold_mask: Optional[torch.Tensor] = None
        self._scaffold_edge_mask: Optional[torch.Tensor] = None
    
    def guided_generate(
        self,
        target_moments: Union[np.ndarray, torch.Tensor],
        num_nodes: Optional[Union[int, List[int], np.ndarray, torch.Tensor]] = None,
        batch_size: int = 32,
        guidance_scale: float = 1.0,
        guidance_start_step: int = 0,
        guidance_end_step: Optional[int] = None,
        num_atom_types: Optional[int] = None,
        J: int = 4,
        num_moments: int = 4,
        labels: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
    ) -> List[str]:
        """
        Generate molecules with scattering moment guidance.
        
        Parameters
        ----------
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
            Timestep index at which to start applying guidance. Can help
            stability by letting early steps establish structure first.
        guidance_end_step : int, optional
            Timestep index at which to stop applying guidance. If None, guidance
            continues until the end. Setting this allows the model to "clean up"
            invalid chemistry in final steps without guidance interference.
        num_atom_types : int, optional
            Number of atom type categories. If None, auto-detected from model's
            dataset_info. Falls back to 10 if not available.
        J : int, default=4
            Number of wavelet scales in scattering transform.
        num_moments : int, default=4
            Number of statistical moments per scattering coefficient.
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
        
        # Auto-detect num_atom_types from model if not explicitly set
        if num_atom_types is None and hasattr(self, 'dataset_info') and self.dataset_info:
            atom_decoder = self.dataset_info.get('atom_decoder', [])
            num_atom_types = len(atom_decoder) if atom_decoder else 10
        elif num_atom_types is None:
            num_atom_types = 10  # fallback default
        
        # Initialize guidance module
        self._guidance = ScatteringMomentGuidance(
            num_atom_types=num_atom_types,
            J=J,
            num_moments=num_moments,
            device=self.device
        )
        
        # Store guidance parameters
        self._target_moments = target_moments
        self._guidance_scale = guidance_scale
        self._guidance_start_step = guidance_start_step
        self._guidance_end_step = guidance_end_step  # None means guide until end
        self._current_step = 0

        # Try to get total steps from the model's diffusion config
        if hasattr(self, 'diffusion_steps'):
            self._total_steps = self.diffusion_steps
        elif hasattr(self, 'noise_schedule') and hasattr(self.noise_schedule, 'timesteps'):
            self._total_steps = self.noise_schedule.timesteps
        else:
            self._total_steps = 500  # default fallback

        print(f"[GUIDANCE] Starting guided generation with:")
        print(f"  - guidance_scale: {guidance_scale}")
        print(f"  - total_steps: {self._total_steps}")
        print(f"  - guidance_window: [{guidance_start_step}, {guidance_end_step or 'end'})")
        print(f"  - target_moments shape: {target_moments.shape}")
        
        # Convert num_nodes to tensor format expected by parent
        if num_nodes is not None and isinstance(num_nodes, int):
            num_nodes = torch.tensor([[num_nodes]] * batch_size, device=self.device)
        
        try:
            # Call parent generate - our overridden sample_p_zs_given_zt will apply guidance
            return super().generate(
                labels=labels,
                num_nodes=num_nodes,
                batch_size=batch_size
            )
        finally:
            # Clean up
            self._guidance = None
            self._target_moments = None
            self._guidance_scale = 1.0
            self._guidance_start_step = 0
            self._guidance_end_step = None
            self._current_step = 0
    
    def guided_generate_with_scaffold(
        self,
        target_moments: Union[np.ndarray, torch.Tensor],
        scaffold_smiles: str,
        scaffold_pattern: Optional[str] = None,
        remove_indices: Optional[List[int]] = None,
        num_nodes: Optional[int] = None,
        batch_size: int = 32,
        guidance_scale: float = 1.0,
        guidance_start_step: int = 0,
        guidance_end_step: Optional[int] = None,
        num_atom_types: Optional[int] = None,
        J: int = 4,
        num_moments: int = 4,
        labels: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
    ) -> List[str]:
        """
        Generate molecules with scattering moment guidance AND scaffold constraints.
        
        The scaffold (core structure) is kept fixed while the rest of the molecule
        is generated with guidance toward target scattering moments.
        
        Parameters
        ----------
        target_moments : array-like
            Target scattering moments [D] or [B, D].
        scaffold_smiles : str
            Full molecule SMILES containing the scaffold.
        scaffold_pattern : str, optional
            SMARTS pattern defining which atoms to KEEP fixed.
            Example: "c1ccccc1" keeps a benzene ring fixed.
        remove_indices : list of int, optional
            Atom indices to REMOVE (regenerate). Atoms NOT in this list are fixed.
            Exactly one of scaffold_pattern or remove_indices must be provided.
        num_nodes : int, optional
            Number of atoms in generated molecule. Defaults to scaffold_smiles atom count.
        batch_size : int, default=32
            Number of molecules to generate.
        guidance_scale : float, default=1.0
            Scale for scattering guidance gradients.
        guidance_start_step : int, default=0
            Timestep to start applying guidance.
        guidance_end_step : int, optional
            Timestep to stop applying guidance. If None, guide until end.
        num_atom_types : int, default=10
            Number of atom types.
        J : int, default=4
            Number of wavelet scales.
        num_moments : int, default=4
            Number of statistical moments.
        labels : array-like, optional
            Property labels for conditional generation.
        
        Returns
        -------
        List[str]
            Generated SMILES strings with scaffold preserved.
        
        Example
        -------
        >>> # Keep benzene ring, regenerate substituents with target moments
        >>> smiles = model.guided_generate_with_scaffold(
        ...     target_moments=target_moments,
        ...     scaffold_smiles="c1ccc(C)cc1N",  # toluene with amine
        ...     scaffold_pattern="c1ccccc1",      # keep benzene
        ...     num_nodes=12,
        ...     batch_size=10
        ... )
        """
        # Get atom decoder from dataset_info
        if not hasattr(self, 'dataset_info') or self.dataset_info is None:
            raise ValueError("Model must be loaded with dataset_info. Call load_from_local first.")
        
        atom_decoder = self.dataset_info.get('atom_decoder')
        bond_decoder = self.dataset_info.get('bond_decoder')
        max_nodes = self.max_node
        
        # Build scaffold tensors
        scaffold_X, scaffold_E, scaffold_mask, node_mask, n_atoms = smiles_to_scaffold(
            full_smiles=scaffold_smiles,
            max_nodes=max_nodes,
            atom_decoder=atom_decoder,
            bond_decoder=bond_decoder,
            scaffold_pattern=scaffold_pattern,
            remove_indices=remove_indices,
            num_nodes=num_nodes,
        )
        
        # Determine output size
        if num_nodes is None:
            num_nodes = n_atoms
        
        scaffold_count = scaffold_mask.sum().item()
        print(f"Scaffold: {scaffold_count} fixed atoms, {num_nodes - scaffold_count} to generate")
        
        # Expand for batch
        self._scaffold_X = scaffold_X.unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        self._scaffold_E = scaffold_E.unsqueeze(0).expand(batch_size, -1, -1, -1).to(self.device)
        self._scaffold_mask = scaffold_mask.unsqueeze(0).expand(batch_size, -1).to(self.device)
        self._scaffold_edge_mask = (
            scaffold_mask.unsqueeze(0) & scaffold_mask.unsqueeze(1)
        ).unsqueeze(0).expand(batch_size, -1, -1).to(self.device)
        
        try:
            # Call guided_generate - scaffold injection happens in sample_p_zs_given_zt
            return self.guided_generate(
                target_moments=target_moments,
                num_nodes=num_nodes,
                batch_size=batch_size,
                guidance_scale=guidance_scale,
                guidance_start_step=guidance_start_step,
                guidance_end_step=guidance_end_step,
                num_atom_types=num_atom_types,
                J=J,
                num_moments=num_moments,
                labels=labels,
            )
        finally:
            # Clean up scaffold
            self._scaffold_X = None
            self._scaffold_E = None
            self._scaffold_mask = None
            self._scaffold_edge_mask = None
    
    def sample_p_zs_given_zt(
        self, s, t, X_t, E_t, properties, node_mask
    ):
        """
        Sample from p(z_s | z_t) with optional scattering moment guidance.
        
        This method overrides the parent to inject gradient-based guidance
        that steers the generation toward target scattering moments.
        """
        bs, n, _ = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)
        
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
        
        # ============== APPLY SCATTERING MOMENT GUIDANCE ==============
        # Guide predictions BEFORE computing posterior (original approach)
        in_guidance_window = (
            self._current_step >= self._guidance_start_step and
            (self._guidance_end_step is None or self._current_step < self._guidance_end_step)
        )

        # Determine if this is a diagnostic step (early, middle, late)
        T = self._total_steps
        diagnostic_steps = [T - 1, T // 2, 1, 0]  # early (high noise), middle, late (low noise)
        is_diagnostic_step = (
            self._diagnostic_logging and
            self._current_step in diagnostic_steps and
            self._guidance is not None
        )

        # Initialize diagnostic variables
        pred_X_before = None
        pred_E_before = None
        pred_X_after_guidance = None
        pred_E_after_guidance = None
        applied_guidance = False

        if (self._guidance is not None and
            self._target_moments is not None and
            in_guidance_window):

            # Save predictions BEFORE guidance for diagnostic comparison
            if is_diagnostic_step:
                pred_X_before = pred_X[0, 0, :].detach().cpu().numpy()
                pred_E_before = pred_E[0, 0, 0, :].detach().cpu().numpy()

            # Compute guidance gradients from predictions
            with torch.enable_grad():
                grad_X, grad_E, loss_val = self._guidance.compute_guidance(
                    pred_X, pred_E, node_mask, self._target_moments, return_loss=True
                )

            # ========== STEP 1A: DIAGNOSTIC LOGGING BEFORE GUIDANCE ==========
            if is_diagnostic_step:
                step = self._current_step

                # --- Prediction magnitudes ---
                pred_X_mean = pred_X.abs().mean().item()
                pred_X_max = pred_X.abs().max().item()
                pred_E_mean = pred_E.abs().mean().item()
                pred_E_max = pred_E.abs().max().item()

                # --- Gradient magnitudes ---
                grad_X_mean = grad_X.abs().mean().item()
                grad_X_max = grad_X.abs().max().item()
                grad_E_mean = grad_E.abs().mean().item()
                grad_E_max = grad_E.abs().max().item()

                # --- The critical ratio: how big is guidance relative to predictions ---
                scale = self._guidance_scale
                ratio_X = (scale * grad_X_mean) / (pred_X_mean + 1e-10)
                ratio_E = (scale * grad_E_mean) / (pred_E_mean + 1e-10)

                # --- Scattering moment distance before guidance ---
                with torch.no_grad():
                    current_moments = self._guidance.compute_moments(pred_X, pred_E, node_mask)
                    moment_dist = (current_moments - self._target_moments).norm(dim=-1).mean().item()

                print(f"\n=== GUIDANCE DIAGNOSTIC at step {step}/{T} ===")
                print(f"  MSE loss to target: {loss_val:.6f}")
                print(f"  pred_X  | mean: {pred_X_mean:.6f} | max: {pred_X_max:.6f}")
                print(f"  grad_X  | mean: {grad_X_mean:.6f} | max: {grad_X_max:.6f}")
                print(f"  ratio_X (scale*grad/pred): {ratio_X:.6f}")
                print(f"  pred_E  | mean: {pred_E_mean:.6f} | max: {pred_E_max:.6f}")
                print(f"  grad_E  | mean: {grad_E_mean:.6f} | max: {grad_E_max:.6f}")
                print(f"  ratio_E (scale*grad/pred): {ratio_E:.6f}")
                print(f"  scattering moment L2 dist to target: {moment_dist:.6f}")
                print(f"==========================================\n")
            # =====================================================================

            # Apply guidance to predictions (then posterior computed from guided pred)
            pred_X = apply_guidance_to_probs(pred_X, grad_X, self._guidance_scale)
            pred_E = apply_guidance_to_probs(pred_E, grad_E, self._guidance_scale)

            # ========== STEP 1B: LOG AFTER GUIDANCE ==========
            if is_diagnostic_step:
                pred_X_after_guidance = pred_X[0, 0, :].detach().cpu().numpy()
                pred_E_after_guidance = pred_E[0, 0, 0, :].detach().cpu().numpy()
            # =================================================

            applied_guidance = True

        self._current_step += 1
        # ==============================================================
        
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

        # ========== STEP 1B (continued): LOG AFTER POSTERIOR ==========
        if applied_guidance and is_diagnostic_step and pred_X_before is not None:

            pred_X_after_posterior = prob_X[0, 0, :].detach().cpu().numpy()
            pred_E_after_posterior = prob_E[0, 0, 0, :].detach().cpu().numpy()

            print(f"\n=== PROBABILITY TRACKING at step {self._current_step - 1}, node 0 ===")
            print(f"  NODES (X):")
            print(f"    Before guidance:  {np.array2string(pred_X_before, precision=4, suppress_small=True)}")
            print(f"    After guidance:   {np.array2string(pred_X_after_guidance, precision=4, suppress_small=True)}")
            print(f"    After posterior:  {np.array2string(pred_X_after_posterior, precision=4, suppress_small=True)}")
            print(f"    Guidance delta:   {np.array2string(pred_X_after_guidance - pred_X_before, precision=4, suppress_small=True)}")
            print(f"    Posterior delta:  {np.array2string(pred_X_after_posterior - pred_X_after_guidance, precision=4, suppress_small=True)}")
            print(f"  EDGES (E) for edge (0,0):")
            print(f"    Before guidance:  {np.array2string(pred_E_before, precision=4, suppress_small=True)}")
            print(f"    After guidance:   {np.array2string(pred_E_after_guidance, precision=4, suppress_small=True)}")
            print(f"    After posterior:  {np.array2string(pred_E_after_posterior, precision=4, suppress_small=True)}")
            print(f"==============================================================\n")
        # ===============================================================

        # Apply classifier-free guidance if configured (from parent)
        if self.guide_scale is not None and self.guide_scale != 1:
            # Get unconditional predictions
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
        if self._guidance is not None:
            # Gumbel-softmax: hard one-hot forward, soft gradients backward
            logits_X = torch.log(prob_X.clamp(min=1e-8))
            logits_E = torch.log(prob_E.clamp(min=1e-8))
            X_s = F.gumbel_softmax(logits_X, tau=0.5, hard=True, dim=-1)
            E_s = F.gumbel_softmax(logits_E, tau=0.5, hard=True, dim=-1)
            # Enforce edge symmetry (take upper triangular, mirror)
            E_idx = E_s.argmax(dim=-1)
            E_idx = torch.triu(E_idx, diagonal=1)
            E_idx = E_idx + E_idx.transpose(1, 2)
            E_s = F.one_hot(E_idx, num_classes=self.input_dim_E).float()
        else:
            # Standard sampling
            sampled_s = sample_discrete_features(prob_X, prob_E, node_mask=node_mask)
            X_s = F.one_hot(sampled_s.X, num_classes=self.input_dim_X).to(self.device).float()
            E_s = F.one_hot(sampled_s.E, num_classes=self.input_dim_E).to(self.device).float()
        
        # ============== INJECT SCAFFOLD CONSTRAINTS ==============
        if self._scaffold_X is not None and self._scaffold_mask is not None:
            # Restore fixed scaffold atoms
            X_s[self._scaffold_mask] = self._scaffold_X[self._scaffold_mask]
            # Restore fixed scaffold edges
            E_s[self._scaffold_edge_mask] = self._scaffold_E[self._scaffold_edge_mask]
            # Ensure symmetry after scaffold injection
            E_s = (E_s + E_s.transpose(1, 2)) / 2
            E_s = (E_s > 0.5).float()  # Re-binarize
        # =========================================================
        
        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)
        
        out_one_hot = PlaceHolder(X=X_s, E=E_s, y=properties)
        
        return out_one_hot.mask(node_mask)
    
    def generate_with_moment_tracking(
        self,
        target_moments: Union[np.ndarray, torch.Tensor],
        num_nodes: Optional[Union[int, List[int], np.ndarray, torch.Tensor]] = None,
        batch_size: int = 32,
        guidance_scale: float = 1.0,
        num_atom_types: int = 10,
        J: int = 4,
        num_moments: int = 4,
        labels: Optional[Union[List, np.ndarray, torch.Tensor]] = None,
    ):
        """
        Generate molecules and track moment distance over diffusion steps.
        
        Returns generated SMILES and trajectory of moment distances.
        Useful for debugging and tuning guidance_scale.
        """
        # This would require more invasive changes to track per-step moments
        # For now, just call guided_generate
        smiles = self.guided_generate(
            target_moments=target_moments,
            num_nodes=num_nodes,
            batch_size=batch_size,
            guidance_scale=guidance_scale,
            num_atom_types=num_atom_types,
            J=J,
            num_moments=num_moments,
            labels=labels,
        )
        return smiles
