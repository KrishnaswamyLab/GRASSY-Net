"""
Evaluation script for comparing guidance approaches.

Compares:
1. Unguided generation (baseline)
2. Direct scattering guidance (existing approach)
3. Classifier guidance (new approach)

Metrics:
- Moment MSE (target vs generated)
- Validity rate
- Uniqueness
- Diversity

Usage:
    python -m guided_diffusion.evaluate_guidance \
        --dit_checkpoint path/to/dit.pt \
        --classifier_checkpoint path/to/classifier.pt \
        --test_smiles_file path/to/test_smiles.txt \
        --num_samples_per_target 10 \
        --guidance_scales 0.1 0.5 1.0 2.0 \
        --output_dir evaluation_results
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional, Dict, Any
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from guided_diffusion.classifier_generator import (
    ClassifierGuidedGraphDIT,
    load_classifier_guided_model,
)
from guided_diffusion.classifier_guidance import ClassifierGuidance
from guided_diffusion.generator import GuidedGraphDIT
from guided_diffusion.guidance import ScatteringMomentGuidance
from models.ScatteringTransform import GraphScatteringTransform
from evaluation.utils import compute_scattering_from_smiles


def load_test_smiles(test_smiles_file: str, max_targets: Optional[int] = None) -> List[str]:
    """Load test SMILES from file."""
    smiles_list = []
    with open(test_smiles_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                smiles_list.append(line)
    
    if max_targets and len(smiles_list) > max_targets:
        indices = np.random.choice(len(smiles_list), max_targets, replace=False)
        smiles_list = [smiles_list[i] for i in indices]
    
    return smiles_list


def compute_validity(smiles_list: List[str]) -> float:
    """Compute fraction of valid SMILES."""
    valid = sum(1 for s in smiles_list if s and Chem.MolFromSmiles(s) is not None)
    return valid / len(smiles_list) if smiles_list else 0.0


def compute_uniqueness(smiles_list: List[str]) -> float:
    """Compute fraction of unique valid SMILES."""
    valid_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) 
                    for s in smiles_list if s and Chem.MolFromSmiles(s)]
    if not valid_smiles:
        return 0.0
    return len(set(valid_smiles)) / len(valid_smiles)


def compute_diversity(smiles_list: List[str]) -> float:
    """Compute internal diversity (1 - avg Tanimoto similarity)."""
    valid_mols = [Chem.MolFromSmiles(s) for s in smiles_list if s and Chem.MolFromSmiles(s)]
    if len(valid_mols) < 2:
        return 0.0
    
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in valid_mols]
    
    similarities = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            similarities.append(sim)
    
    return 1.0 - np.mean(similarities) if similarities else 0.0


def compute_similarity_to_target(smiles_list: List[str], target_smiles: str) -> float:
    """Compute average Tanimoto similarity to target molecule."""
    target_mol = Chem.MolFromSmiles(target_smiles)
    if target_mol is None:
        return 0.0
    
    target_fp = AllChem.GetMorganFingerprintAsBitVect(target_mol, 2, 2048)
    
    similarities = []
    for smi in smiles_list:
        if smi:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
                sim = DataStructs.TanimotoSimilarity(target_fp, fp)
                similarities.append(sim)
    
    return np.mean(similarities) if similarities else 0.0


def evaluate_unguided(
    model,
    target_moments: torch.Tensor,
    num_samples: int,
    num_nodes: Optional[int],
    scattering_model,
    atom_types: List[str],
    device: str,
) -> Dict[str, Any]:
    """Evaluate unguided generation."""
    # Generate
    num_nodes_tensor = None
    if num_nodes is not None:
        num_nodes_tensor = torch.tensor([[num_nodes]] * num_samples, device=device)
    
    smiles_list = model.generate(
        num_nodes=num_nodes_tensor,
        batch_size=num_samples,
    )
    
    # Compute metrics
    validity = compute_validity(smiles_list)
    uniqueness = compute_uniqueness(smiles_list)
    diversity = compute_diversity(smiles_list)
    
    # Compute moment distances
    valid_smiles = [s for s in smiles_list if s and Chem.MolFromSmiles(s)]
    if valid_smiles:
        gen_moments = compute_scattering_from_smiles(
            valid_smiles, scattering_model, atom_types, device
        )
        target_expanded = target_moments.unsqueeze(0).expand(gen_moments.shape[0], -1)
        distances = ((gen_moments.to(device) - target_expanded) ** 2).mean(dim=-1)
        moment_mse = distances.mean().item()
        moment_mse_min = distances.min().item()
    else:
        moment_mse = float('inf')
        moment_mse_min = float('inf')
    
    return {
        'validity': validity,
        'uniqueness': uniqueness,
        'diversity': diversity,
        'moment_mse': moment_mse,
        'moment_mse_min': moment_mse_min,
        'num_valid': len(valid_smiles),
        'smiles': valid_smiles[:10],  # Save sample
    }


def evaluate_direct_guidance(
    model: GuidedGraphDIT,
    target_moments: torch.Tensor,
    num_samples: int,
    num_nodes: Optional[int],
    guidance_scale: float,
    num_atom_types: int,
    J: int,
    num_moments: int,
    scattering_model,
    atom_types: List[str],
    device: str,
) -> Dict[str, Any]:
    """Evaluate direct scattering guidance."""
    # Generate with guidance
    smiles_list = model.guided_generate(
        target_moments=target_moments,
        num_nodes=num_nodes,
        batch_size=num_samples,
        guidance_scale=guidance_scale,
        num_atom_types=num_atom_types,
        J=J,
        num_moments=num_moments,
    )
    
    # Compute metrics
    validity = compute_validity(smiles_list)
    uniqueness = compute_uniqueness(smiles_list)
    diversity = compute_diversity(smiles_list)
    
    # Compute moment distances
    valid_smiles = [s for s in smiles_list if s and Chem.MolFromSmiles(s)]
    if valid_smiles:
        gen_moments = compute_scattering_from_smiles(
            valid_smiles, scattering_model, atom_types, device
        )
        target_expanded = target_moments.unsqueeze(0).expand(gen_moments.shape[0], -1)
        distances = ((gen_moments.to(device) - target_expanded) ** 2).mean(dim=-1)
        moment_mse = distances.mean().item()
        moment_mse_min = distances.min().item()
    else:
        moment_mse = float('inf')
        moment_mse_min = float('inf')
    
    return {
        'validity': validity,
        'uniqueness': uniqueness,
        'diversity': diversity,
        'moment_mse': moment_mse,
        'moment_mse_min': moment_mse_min,
        'num_valid': len(valid_smiles),
        'smiles': valid_smiles[:10],
    }


def evaluate_classifier_guidance(
    model: ClassifierGuidedGraphDIT,
    classifier_guidance: ClassifierGuidance,
    target_moments: torch.Tensor,
    num_samples: int,
    num_nodes: Optional[int],
    guidance_scale: float,
    scattering_model,
    atom_types: List[str],
    device: str,
) -> Dict[str, Any]:
    """Evaluate classifier-based guidance."""
    # Generate with guidance
    smiles_list = model.classifier_guided_generate_with_preloaded(
        classifier_guidance=classifier_guidance,
        target_moments=target_moments,
        num_nodes=num_nodes,
        batch_size=num_samples,
        guidance_scale=guidance_scale,
    )
    
    # Compute metrics
    validity = compute_validity(smiles_list)
    uniqueness = compute_uniqueness(smiles_list)
    diversity = compute_diversity(smiles_list)
    
    # Compute moment distances
    valid_smiles = [s for s in smiles_list if s and Chem.MolFromSmiles(s)]
    if valid_smiles:
        gen_moments = compute_scattering_from_smiles(
            valid_smiles, scattering_model, atom_types, device
        )
        target_expanded = target_moments.unsqueeze(0).expand(gen_moments.shape[0], -1)
        distances = ((gen_moments.to(device) - target_expanded) ** 2).mean(dim=-1)
        moment_mse = distances.mean().item()
        moment_mse_min = distances.min().item()
    else:
        moment_mse = float('inf')
        moment_mse_min = float('inf')
    
    return {
        'validity': validity,
        'uniqueness': uniqueness,
        'diversity': diversity,
        'moment_mse': moment_mse,
        'moment_mse_min': moment_mse_min,
        'num_valid': len(valid_smiles),
        'smiles': valid_smiles[:10],
    }


def run_evaluation(
    dit_checkpoint: str,
    classifier_checkpoint: Optional[str],
    test_smiles_file: str,
    output_dir: str,
    num_samples_per_target: int = 10,
    guidance_scales: List[float] = [0.1, 0.5, 1.0, 2.0],
    max_targets: Optional[int] = None,
    num_nodes: Optional[int] = None,
    J: int = 4,
    num_moments: int = 4,
    device: str = 'auto',
    skip_direct: bool = False,
    skip_classifier: bool = False,
):
    """
    Run comprehensive evaluation of guidance approaches.
    
    Args:
        dit_checkpoint: Path to DiT model checkpoint
        classifier_checkpoint: Path to classifier checkpoint (None = skip classifier guidance)
        test_smiles_file: File with test SMILES (one per line)
        output_dir: Directory to save results
        num_samples_per_target: Number of samples to generate per target molecule
        guidance_scales: List of guidance scales to test
        max_targets: Maximum number of target molecules to test
        num_nodes: Number of atoms per molecule (None = auto from target)
        J: Number of wavelet scales
        num_moments: Number of statistical moments
        device: Device to use
        skip_direct: Skip direct scattering guidance evaluation
        skip_classifier: Skip classifier guidance evaluation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Device setup
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load test SMILES
    print(f"\nLoading test molecules from {test_smiles_file}...")
    test_smiles = load_test_smiles(test_smiles_file, max_targets)
    print(f"Loaded {len(test_smiles)} test molecules")
    
    # Load DiT model (as both GuidedGraphDIT and ClassifierGuidedGraphDIT)
    print(f"\nLoading DiT model from {dit_checkpoint}...")
    
    # For direct guidance
    if not skip_direct:
        direct_model = GuidedGraphDIT()
        direct_model.load_from_local(dit_checkpoint)
        direct_model.device = device
    
    # For classifier guidance  
    if not skip_classifier and classifier_checkpoint:
        classifier_model = load_classifier_guided_model(dit_checkpoint, str(device))
        classifier_guidance = ClassifierGuidance(classifier_checkpoint, str(device))
    
    # Get atom decoder from model
    atom_decoder = None
    ref_model = direct_model if not skip_direct else classifier_model
    if hasattr(ref_model, 'dataset_info') and ref_model.dataset_info:
        atom_decoder = ref_model.dataset_info.get('atom_decoder', None)
    
    if atom_decoder is None:
        atom_decoder = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B"]
    
    num_atom_types = len(atom_decoder)
    print(f"Atom types ({num_atom_types}): {atom_decoder}")
    
    # Initialize scattering model for evaluation
    scattering_model = GraphScatteringTransform(
        in_channels=num_atom_types, J=J, num_moments=num_moments
    ).to(device)
    scattering_model.eval()
    
    # Results storage
    all_results = {
        'config': {
            'dit_checkpoint': dit_checkpoint,
            'classifier_checkpoint': classifier_checkpoint,
            'test_smiles_file': test_smiles_file,
            'num_samples_per_target': num_samples_per_target,
            'guidance_scales': guidance_scales,
            'num_targets': len(test_smiles),
            'num_atom_types': num_atom_types,
            'J': J,
            'num_moments': num_moments,
        },
        'per_target': [],
        'aggregated': {},
    }
    
    # Evaluate each target
    for target_idx, target_smi in enumerate(tqdm(test_smiles, desc="Evaluating targets")):
        target_mol = Chem.MolFromSmiles(target_smi)
        if target_mol is None:
            continue
        
        target_num_nodes = num_nodes if num_nodes else target_mol.GetNumAtoms()
        
        # Compute target moments
        target_moments = compute_scattering_from_smiles(
            [target_smi], scattering_model, atom_decoder, str(device)
        )
        if target_moments.shape[0] == 0:
            continue
        target_moments = target_moments[0].to(device)
        
        target_results = {
            'target_smiles': target_smi,
            'target_num_atoms': target_num_nodes,
            'unguided': None,
            'direct': {},
            'classifier': {},
        }
        
        # Evaluate unguided (only once per target)
        unguided_result = evaluate_unguided(
            ref_model, target_moments, num_samples_per_target, target_num_nodes,
            scattering_model, atom_decoder, str(device)
        )
        target_results['unguided'] = unguided_result
        
        # Evaluate guidance methods at each scale
        for scale in guidance_scales:
            # Direct scattering guidance
            if not skip_direct:
                direct_result = evaluate_direct_guidance(
                    direct_model, target_moments, num_samples_per_target, target_num_nodes,
                    scale, num_atom_types, J, num_moments,
                    scattering_model, atom_decoder, str(device)
                )
                target_results['direct'][str(scale)] = direct_result
            
            # Classifier guidance
            if not skip_classifier and classifier_checkpoint:
                clf_result = evaluate_classifier_guidance(
                    classifier_model, classifier_guidance, target_moments,
                    num_samples_per_target, target_num_nodes, scale,
                    scattering_model, atom_decoder, str(device)
                )
                target_results['classifier'][str(scale)] = clf_result
        
        all_results['per_target'].append(target_results)
    
    # Aggregate results
    print("\nAggregating results...")
    aggregate_results(all_results)
    
    # Save results
    results_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")
    
    # Print summary
    print_summary(all_results)
    
    return all_results


def aggregate_results(results: Dict[str, Any]):
    """Compute aggregated statistics from per-target results."""
    per_target = results['per_target']
    if not per_target:
        return
    
    guidance_scales = results['config']['guidance_scales']
    
    # Initialize aggregation
    aggregated = {
        'unguided': {'validity': [], 'uniqueness': [], 'diversity': [], 'moment_mse': [], 'moment_mse_min': []},
        'direct': {str(s): {'validity': [], 'uniqueness': [], 'diversity': [], 'moment_mse': [], 'moment_mse_min': []} for s in guidance_scales},
        'classifier': {str(s): {'validity': [], 'uniqueness': [], 'diversity': [], 'moment_mse': [], 'moment_mse_min': []} for s in guidance_scales},
    }
    
    for target in per_target:
        if target['unguided']:
            for metric in ['validity', 'uniqueness', 'diversity', 'moment_mse', 'moment_mse_min']:
                aggregated['unguided'][metric].append(target['unguided'][metric])
        
        for scale in guidance_scales:
            scale_str = str(scale)
            if scale_str in target['direct']:
                for metric in ['validity', 'uniqueness', 'diversity', 'moment_mse', 'moment_mse_min']:
                    aggregated['direct'][scale_str][metric].append(target['direct'][scale_str][metric])
            
            if scale_str in target['classifier']:
                for metric in ['validity', 'uniqueness', 'diversity', 'moment_mse', 'moment_mse_min']:
                    aggregated['classifier'][scale_str][metric].append(target['classifier'][scale_str][metric])
    
    # Compute means
    def mean_or_nan(lst):
        return np.mean(lst) if lst else float('nan')
    
    results['aggregated']['unguided'] = {
        metric: mean_or_nan(aggregated['unguided'][metric])
        for metric in ['validity', 'uniqueness', 'diversity', 'moment_mse', 'moment_mse_min']
    }
    
    results['aggregated']['direct'] = {}
    results['aggregated']['classifier'] = {}
    
    for scale in guidance_scales:
        scale_str = str(scale)
        results['aggregated']['direct'][scale_str] = {
            metric: mean_or_nan(aggregated['direct'][scale_str][metric])
            for metric in ['validity', 'uniqueness', 'diversity', 'moment_mse', 'moment_mse_min']
        }
        results['aggregated']['classifier'][scale_str] = {
            metric: mean_or_nan(aggregated['classifier'][scale_str][metric])
            for metric in ['validity', 'uniqueness', 'diversity', 'moment_mse', 'moment_mse_min']
        }


def print_summary(results: Dict[str, Any]):
    """Print summary table of results."""
    agg = results['aggregated']
    scales = results['config']['guidance_scales']
    
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    # Header
    print(f"\n{'Method':<25} {'Scale':<8} {'Validity':<10} {'Unique':<10} {'Diversity':<10} {'MSE':<12} {'MSE_min':<10}")
    print("-" * 85)
    
    # Unguided baseline
    if 'unguided' in agg:
        u = agg['unguided']
        print(f"{'Unguided':<25} {'-':<8} {u['validity']:.3f}     {u['uniqueness']:.3f}     {u['diversity']:.3f}     {u['moment_mse']:.4f}     {u['moment_mse_min']:.4f}")
    
    print("-" * 85)
    
    # Direct guidance
    if 'direct' in agg and agg['direct']:
        for scale in scales:
            d = agg['direct'].get(str(scale), {})
            if d:
                print(f"{'Direct Scattering':<25} {scale:<8} {d.get('validity', 0):.3f}     {d.get('uniqueness', 0):.3f}     {d.get('diversity', 0):.3f}     {d.get('moment_mse', 0):.4f}     {d.get('moment_mse_min', 0):.4f}")
    
    print("-" * 85)
    
    # Classifier guidance
    if 'classifier' in agg and agg['classifier']:
        for scale in scales:
            c = agg['classifier'].get(str(scale), {})
            if c:
                print(f"{'Classifier':<25} {scale:<8} {c.get('validity', 0):.3f}     {c.get('uniqueness', 0):.3f}     {c.get('diversity', 0):.3f}     {c.get('moment_mse', 0):.4f}     {c.get('moment_mse_min', 0):.4f}")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate and compare guidance approaches',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Model arguments
    parser.add_argument('--dit_checkpoint', required=True,
                        help='Path to DiT model checkpoint')
    parser.add_argument('--classifier_checkpoint', default=None,
                        help='Path to classifier checkpoint (None = skip classifier guidance)')
    
    # Data arguments
    parser.add_argument('--test_smiles_file', required=True,
                        help='File with test SMILES (one per line)')
    parser.add_argument('--output_dir', default='evaluation_results',
                        help='Output directory for results')
    
    # Evaluation arguments
    parser.add_argument('--num_samples_per_target', type=int, default=10,
                        help='Number of samples per target molecule')
    parser.add_argument('--guidance_scales', type=float, nargs='+',
                        default=[0.1, 0.5, 1.0, 2.0],
                        help='Guidance scales to test')
    parser.add_argument('--max_targets', type=int, default=None,
                        help='Maximum number of target molecules')
    parser.add_argument('--num_nodes', type=int, default=None,
                        help='Number of atoms per molecule (None = auto)')
    
    # Scattering parameters
    parser.add_argument('--J', type=int, default=4,
                        help='Number of wavelet scales')
    parser.add_argument('--num_moments', type=int, default=4,
                        help='Number of statistical moments')
    
    # Flags
    parser.add_argument('--skip_direct', action='store_true',
                        help='Skip direct scattering guidance evaluation')
    parser.add_argument('--skip_classifier', action='store_true',
                        help='Skip classifier guidance evaluation')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use')
    
    args = parser.parse_args()
    
    run_evaluation(
        dit_checkpoint=args.dit_checkpoint,
        classifier_checkpoint=args.classifier_checkpoint,
        test_smiles_file=args.test_smiles_file,
        output_dir=args.output_dir,
        num_samples_per_target=args.num_samples_per_target,
        guidance_scales=args.guidance_scales,
        max_targets=args.max_targets,
        num_nodes=args.num_nodes,
        J=args.J,
        num_moments=args.num_moments,
        device=args.device,
        skip_direct=args.skip_direct,
        skip_classifier=args.skip_classifier,
    )


if __name__ == '__main__':
    main()
