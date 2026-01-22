"""
BACE Benchmark Evaluation Script for GRASSY-DiT.

Evaluates GRASSY-DiT molecular generation on BACE dataset using metrics
from Graph DiT paper Table 2: Validity, Coverage, Diversity, Similarity, FCD.

Usage:
    python -m evaluation.bace_eval \
        --checkpoint grassy_dit/checkpoints/bace/checkpoint_best.pt \
        --test-smiles grassy_dit/data_bace/molecules.csv \
        --test-scattering grassy_dit/data_bace/scattering_moments.npy \
        --splits datasets/BACE_splits.json \
        --num-samples 10000
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "moses"))

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

from evaluation.utils import load_dit_model

# Output directory
RESULTS_DIR = Path(__file__).parent / "results"

# Baseline values from Graph DiT paper Table 2
BACE_BASELINES = {
    "Graph DiT": {
        "validity": 0.8674,
        "validity_filtered": 0.8495,
        "coverage": "8/8",
        "diversity": 0.8238,
        "similarity": 0.8752,
        "fcd": 7.0456,
    },
    "DiGress": {
        "validity": 0.3511,
        "validity_filtered": 0.2858,
        "coverage": "8/8",
        "diversity": 0.8862,
        "similarity": 0.6942,
        "fcd": 24.6560,
    },
    "MOOD": {
        "validity": 0.9947,
        "validity_filtered": 0.4502,
        "coverage": "8/8",
        "diversity": 0.8902,
        "similarity": 0.2587,
        "fcd": 44.2394,
    },
}

# Reference heavy atom types (8 types as per Graph DiT paper)
REFERENCE_ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I']


class BACEBenchmark:
    """
    BACE benchmark evaluation for GRASSY-DiT.
    
    Computes metrics matching Graph DiT paper Table 2.
    """

    def __init__(
        self,
        checkpoint: str,
        test_smiles_path: str,
        test_scattering_path: str,
        splits_path: Optional[str] = None,
        device: str = "cuda",
        n_jobs: int = 4,
    ):
        """
        Initialize the benchmark.

        Args:
            checkpoint: Path to trained GRASSY-DiT checkpoint
            test_smiles_path: Path to molecules.csv with SMILES
            test_scattering_path: Path to scattering_moments.npy
            splits_path: Path to BACE_splits.json (to filter test set)
            device: Computation device
            n_jobs: Number of workers for metric computation
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.n_jobs = n_jobs

        # Load model
        print(f"Loading DiT model from {checkpoint}...")
        self.model = load_dit_model(checkpoint, self.device)

        # Load test data
        print(f"Loading test data...")
        import pandas as pd
        
        all_smiles = pd.read_csv(test_smiles_path)['smiles'].tolist()
        all_scattering = np.load(test_scattering_path)
        
        # Filter to test set if splits provided
        if splits_path and os.path.exists(splits_path):
            with open(splits_path, 'r') as f:
                splits = json.load(f)
            test_indices = splits.get('test', list(range(len(all_smiles))))
            self.test_smiles = [all_smiles[i] for i in test_indices if i < len(all_smiles)]
            self.test_scattering = all_scattering[test_indices]
            print(f"Using test split: {len(self.test_smiles)} molecules")
        else:
            self.test_smiles = all_smiles
            self.test_scattering = all_scattering
            print(f"Using all data: {len(self.test_smiles)} molecules")

    def generate_samples(
        self,
        num_samples: int = 10000,
        batch_size: int = 64,
    ) -> List[str]:
        """
        Generate molecules conditioned on test set scattering moments.

        Args:
            num_samples: Total number of molecules to generate
            batch_size: Generation batch size

        Returns:
            List of generated SMILES strings
        """
        print(f"\nGenerating {num_samples} molecules...")
        
        all_smiles = []
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(num_batches), desc="Generating"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, num_samples)
            current_batch_size = batch_end - batch_start
            
            # Sample scattering from test set (with replacement)
            indices = np.random.choice(len(self.test_scattering), current_batch_size, replace=True)
            batch_scattering = torch.tensor(
                self.test_scattering[indices], 
                dtype=torch.float32,
                device=self.device
            )
            
            # Generate
            smiles_batch = self.model.generate(
                scattering=batch_scattering,
                num_nodes=None,
                batch_size=current_batch_size,
            )
            
            # Filter None values
            valid_smiles = [s for s in smiles_batch if s is not None]
            all_smiles.extend(valid_smiles)

        print(f"Generated {len(all_smiles)} molecules")
        return all_smiles

    def compute_validity(self, smiles_list: List[str]) -> Dict[str, float]:
        """
        Compute validity metrics.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dict with 'validity' and 'validity_filtered'
        """
        from moses.metrics import fraction_valid, fraction_passes_filters
        
        # Basic validity (RDKit can parse)
        valid_mols = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_mols.append(mol)
        
        validity = len(valid_mols) / len(smiles_list) if smiles_list else 0.0
        
        # Filtered validity (passes medicinal chemistry filters)
        valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
        if valid_smiles:
            validity_filtered = fraction_passes_filters(valid_smiles, n_jobs=self.n_jobs)
        else:
            validity_filtered = 0.0
        
        return {
            'validity': validity,
            'validity_filtered': validity_filtered,
            'n_valid': len(valid_mols),
        }

    def compute_coverage(self, smiles_list: List[str]) -> Dict[str, any]:
        """
        Compute atom type coverage.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Dict with coverage info
        """
        found_atoms: Set[str] = set()
        
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol()
                    if symbol in REFERENCE_ATOM_TYPES:
                        found_atoms.add(symbol)
        
        coverage_count = len(found_atoms)
        coverage_str = f"{coverage_count}/{len(REFERENCE_ATOM_TYPES)}"
        
        return {
            'coverage': coverage_str,
            'coverage_count': coverage_count,
            'coverage_total': len(REFERENCE_ATOM_TYPES),
            'found_atoms': sorted(list(found_atoms)),
            'missing_atoms': sorted(list(set(REFERENCE_ATOM_TYPES) - found_atoms)),
        }

    def compute_diversity(self, smiles_list: List[str]) -> float:
        """
        Compute internal diversity (1 - average pairwise Tanimoto similarity).

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Internal diversity score
        """
        from moses.metrics import internal_diversity
        
        # Get valid molecules
        valid_mols = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_mols.append(mol)
        
        if len(valid_mols) < 2:
            return 0.0
        
        diversity = internal_diversity(valid_mols, n_jobs=self.n_jobs, device=self.device)
        return float(diversity)

    def compute_similarity(self, smiles_list: List[str]) -> float:
        """
        Compute fragment similarity to test set.

        Args:
            smiles_list: List of generated SMILES

        Returns:
            Fragment similarity score
        """
        from moses.metrics import FragMetric
        
        # Get valid generated molecules
        gen_mols = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                gen_mols.append(mol)
        
        if not gen_mols:
            return 0.0
        
        # Get reference molecules (test set)
        ref_mols = []
        for smi in self.test_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                ref_mols.append(mol)
        
        if not ref_mols:
            return 0.0
        
        # Compute fragment similarity
        metric = FragMetric(n_jobs=self.n_jobs)
        pref = metric.precalc(ref_mols)
        similarity = metric(gen=gen_mols, pref=pref)
        
        return float(similarity)

    def compute_fcd(self, smiles_list: List[str]) -> float:
        """
        Compute Frechet ChemNet Distance.

        Args:
            smiles_list: List of generated SMILES

        Returns:
            FCD score (lower is better)
        """
        from fcd_torch import FCD as FCDMetric
        
        # Get valid generated SMILES
        gen_smiles = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                gen_smiles.append(Chem.MolToSmiles(mol))
        
        if not gen_smiles:
            return float('inf')
        
        # Compute FCD
        fcd_metric = FCDMetric(n_jobs=self.n_jobs, device=self.device)
        
        # Precalculate for reference set
        ref_smiles = [smi for smi in self.test_smiles if Chem.MolFromSmiles(smi) is not None]
        pref = fcd_metric.precalc(ref_smiles)
        
        # Compute FCD
        fcd_score = fcd_metric(gen=gen_smiles, pref=pref)
        
        return float(fcd_score)

    def compute_all_metrics(self, smiles_list: List[str]) -> Dict[str, any]:
        """
        Compute all metrics.

        Args:
            smiles_list: List of generated SMILES

        Returns:
            Dictionary with all metrics
        """
        print("\nComputing metrics...")
        
        metrics = {}
        
        # Validity
        print("  Computing validity...")
        validity_metrics = self.compute_validity(smiles_list)
        metrics.update(validity_metrics)
        
        # Coverage
        print("  Computing coverage...")
        coverage_metrics = self.compute_coverage(smiles_list)
        metrics['coverage'] = coverage_metrics['coverage']
        metrics['coverage_details'] = coverage_metrics
        
        # Diversity
        print("  Computing diversity...")
        metrics['diversity'] = self.compute_diversity(smiles_list)
        
        # Similarity
        print("  Computing similarity...")
        metrics['similarity'] = self.compute_similarity(smiles_list)
        
        # FCD
        print("  Computing FCD...")
        metrics['fcd'] = self.compute_fcd(smiles_list)
        
        return metrics

    def format_comparison_table(self, metrics: Dict[str, any]) -> str:
        """
        Format comparison table with baselines.

        Args:
            metrics: Computed metrics for GRASSY-DiT

        Returns:
            Formatted markdown table
        """
        # Header
        header = "| Model | Validity | Coverage | Diversity | Similarity | Distance |"
        separator = "|-------|----------|----------|-----------|------------|----------|"
        
        rows = [header, separator]
        
        # Add baselines
        for model_name, baseline in BACE_BASELINES.items():
            validity_str = f"{baseline['validity']:.4f} ({baseline['validity_filtered']:.4f})"
            row = f"| {model_name} | {validity_str} | {baseline['coverage']} | {baseline['diversity']:.4f} | {baseline['similarity']:.4f} | {baseline['fcd']:.4f} |"
            rows.append(row)
        
        # Add GRASSY-DiT
        validity_str = f"{metrics['validity']:.4f} ({metrics['validity_filtered']:.4f})"
        row = f"| GRASSY-DiT | {validity_str} | {metrics['coverage']} | {metrics['diversity']:.4f} | {metrics['similarity']:.4f} | {metrics['fcd']:.4f} |"
        rows.append(row)
        
        return "\n".join(rows)

    def run(
        self,
        num_samples: int = 10000,
        batch_size: int = 64,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Run the full benchmark evaluation.

        Args:
            num_samples: Number of molecules to generate
            batch_size: Generation batch size
            output_dir: Output directory for results

        Returns:
            Dictionary containing all results
        """
        if output_dir is None:
            output_dir = RESULTS_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Generate samples
        generated_smiles = self.generate_samples(
            num_samples=num_samples,
            batch_size=batch_size,
        )

        # Compute metrics
        metrics = self.compute_all_metrics(generated_smiles)

        # Save generated samples
        samples_path = output_dir / f"bace_generated_{timestamp}.txt"
        with open(samples_path, "w") as f:
            for smi in generated_smiles:
                f.write(smi + "\n")
        print(f"\nSaved generated samples to {samples_path}")

        # Create comparison table
        comparison_table = self.format_comparison_table(metrics)

        # Print results
        print("\n" + "=" * 80)
        print("BACE Benchmark Results")
        print("=" * 80)
        print(f"\nGenerated: {len(generated_smiles)} molecules")
        print(f"Valid: {metrics['n_valid']} ({metrics['validity']:.2%})")
        print(f"\nMetrics:")
        print(f"  Validity: {metrics['validity']:.4f} ({metrics['validity_filtered']:.4f} filtered)")
        print(f"  Coverage: {metrics['coverage']}")
        print(f"  Diversity: {metrics['diversity']:.4f}")
        print(f"  Similarity: {metrics['similarity']:.4f}")
        print(f"  FCD: {metrics['fcd']:.4f}")
        print(f"\nComparison with baselines:")
        print(comparison_table)

        # Save results
        results = {
            "config": {
                "num_samples": num_samples,
                "batch_size": batch_size,
                "timestamp": timestamp,
            },
            "metrics": metrics,
            "baselines": BACE_BASELINES,
            "comparison_table": comparison_table,
        }

        results_path = output_dir / f"bace_metrics_{timestamp}.json"
        
        # Convert numpy types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            return obj

        with open(results_path, "w") as f:
            json.dump(convert_to_native(results), f, indent=2)
        print(f"\nSaved results to {results_path}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="BACE Benchmark Evaluation for GRASSY-DiT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python -m evaluation.bace_eval \\
      --checkpoint grassy_dit/checkpoints/bace/checkpoint_best.pt \\
      --test-smiles grassy_dit/data_bace/molecules.csv \\
      --test-scattering grassy_dit/data_bace/scattering_moments.npy

  # With test split and more samples
  python -m evaluation.bace_eval \\
      --checkpoint grassy_dit/checkpoints/bace/checkpoint_best.pt \\
      --test-smiles grassy_dit/data_bace/molecules.csv \\
      --test-scattering grassy_dit/data_bace/scattering_moments.npy \\
      --splits datasets/BACE_splits.json \\
      --num-samples 10000
        """,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained GRASSY-DiT model checkpoint",
    )
    parser.add_argument(
        "--test-smiles",
        required=True,
        help="Path to molecules.csv with test SMILES",
    )
    parser.add_argument(
        "--test-scattering",
        required=True,
        help="Path to scattering_moments.npy",
    )

    # Optional arguments
    parser.add_argument(
        "--splits",
        default=None,
        help="Path to BACE_splits.json (to filter test set)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of molecules to generate (default: 10000)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Generation batch size (default: 64)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: evaluation/results/)",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Computation device (default: cuda)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of workers for metric computation (default: 4)",
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = BACEBenchmark(
        checkpoint=args.checkpoint,
        test_smiles_path=args.test_smiles,
        test_scattering_path=args.test_scattering,
        splits_path=args.splits,
        device=args.device,
        n_jobs=args.n_jobs,
    )

    results = benchmark.run(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
