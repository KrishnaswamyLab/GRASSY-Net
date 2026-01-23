"""
BACE Benchmark Evaluation Script for GRASSY-DiT.

Evaluates GRASSY-DiT molecular generation on BACE dataset using metrics
from Graph DiT paper Table 2: Validity, Coverage, Diversity, Similarity, FCD.
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

from grassy_dit.sample import load_model_from_checkpoint

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
REFERENCE_ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I"]


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
        config_path: Optional[str] = None,
        smiles_col: str = "smiles",
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.n_jobs = n_jobs

        # Load model
        print(f"Loading DiT model from {checkpoint}...")
        self.model = load_model_from_checkpoint(
            checkpoint_path=checkpoint,
            config_path=config_path,
            device=self.device,
            scattering_path=test_scattering_path,
        )

        # Load test data
        print("Loading test data...")
        import pandas as pd

        all_smiles = pd.read_csv(test_smiles_path)[smiles_col].tolist()
        all_scattering = np.load(test_scattering_path)

        self.test_smiles = all_smiles
        self.test_scattering = all_scattering
        print(f"Using test files directly: {len(self.test_smiles)} molecules")

    def generate_samples(
        self,
        num_samples: int = 10000,
        batch_size: int = 64,
    ) -> List[str]:
        """
        Generate molecules conditioned on test set scattering moments.
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
                device=self.device,
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

    def score_single_molecule_similarity(
        self,
        index: int,
        num_samples: int = 1,
    ) -> Dict[str, float]:
        """
        Take a test molecule, condition on its scattering, and score similarity to generated samples.
        """
        if index < 0 or index >= len(self.test_smiles):
            raise IndexError(f"Index {index} out of range for test set of size {len(self.test_smiles)}")

        ref_smiles = self.test_smiles[index]
        ref_mol = Chem.MolFromSmiles(ref_smiles)
        if ref_mol is None:
            raise ValueError(f"Invalid reference SMILES at index {index}: {ref_smiles}")

        scattering = self.test_scattering[index]
        scattering_batch = np.repeat(scattering[None, :], num_samples, axis=0)
        batch_scattering = torch.tensor(scattering_batch, dtype=torch.float32, device=self.device)

        smiles_batch = self.model.generate(
            scattering=batch_scattering,
            num_nodes=None,
            batch_size=num_samples,
        )
        valid_smiles = [s for s in smiles_batch if s is not None]
        gen_mols = [Chem.MolFromSmiles(s) for s in valid_smiles]
        gen_mols = [m for m in gen_mols if m is not None]

        # Morgan Tanimoto
        tanimoto_scores = []
        ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
        for mol in gen_mols:
            gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            tanimoto_scores.append(DataStructs.TanimotoSimilarity(ref_fp, gen_fp))

        tanimoto_mean = float(np.mean(tanimoto_scores)) if tanimoto_scores else 0.0
        tanimoto_max = float(np.max(tanimoto_scores)) if tanimoto_scores else 0.0

        # MOSES FragMetric similarity
        frag_similarity = 0.0
        if gen_mols:
            from moses.metrics import FragMetric

            metric = FragMetric(n_jobs=self.n_jobs)
            pref = metric.precalc([ref_mol])
            frag_similarity = float(metric(gen=gen_mols, pref=pref))

        return {
            "reference_smiles": ref_smiles,
            "generated_count": len(gen_mols),
            "tanimoto_mean": tanimoto_mean,
            "tanimoto_max": tanimoto_max,
            "frag_similarity": frag_similarity,
        }

    def compute_validity(self, smiles_list: List[str]) -> Dict[str, float]:
        from moses.metrics import fraction_passes_filters

        valid_mols = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                valid_mols.append(mol)

        validity = len(valid_mols) / len(smiles_list) if smiles_list else 0.0

        valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
        if valid_smiles:
            validity_filtered = fraction_passes_filters(valid_smiles, n_jobs=self.n_jobs)
        else:
            validity_filtered = 0.0

        return {
            "validity": validity,
            "validity_filtered": validity_filtered,
            "n_valid": len(valid_mols),
        }

    def compute_coverage(self, smiles_list: List[str]) -> Dict[str, any]:
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
            "coverage": coverage_str,
            "coverage_count": coverage_count,
            "coverage_total": len(REFERENCE_ATOM_TYPES),
            "found_atoms": sorted(list(found_atoms)),
            "missing_atoms": sorted(list(set(REFERENCE_ATOM_TYPES) - found_atoms)),
        }

    def compute_diversity(self, smiles_list: List[str]) -> float:
        from moses.metrics import internal_diversity

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
        from moses.metrics import FragMetric

        gen_mols = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                gen_mols.append(mol)

        if not gen_mols:
            return 0.0

        ref_mols = []
        for smi in self.test_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                ref_mols.append(mol)

        if not ref_mols:
            return 0.0

        metric = FragMetric(n_jobs=self.n_jobs)
        pref = metric.precalc(ref_mols)
        similarity = metric(gen=gen_mols, pref=pref)

        return float(similarity)

    def compute_fcd(self, smiles_list: List[str]) -> float:
        from fcd_torch import FCD as FCDMetric

        gen_smiles = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                gen_smiles.append(Chem.MolToSmiles(mol))

        if not gen_smiles:
            return float("inf")

        fcd_metric = FCDMetric(n_jobs=self.n_jobs, device=self.device)

        ref_smiles = [smi for smi in self.test_smiles if Chem.MolFromSmiles(smi) is not None]
        pref = fcd_metric.precalc(ref_smiles)

        fcd_score = fcd_metric(gen=gen_smiles, pref=pref)

        return float(fcd_score)

    def compute_all_metrics(self, smiles_list: List[str]) -> Dict[str, any]:
        print("\nComputing metrics...")

        metrics = {}

        print("  Computing validity...")
        validity_metrics = self.compute_validity(smiles_list)
        metrics.update(validity_metrics)

        print("  Computing coverage...")
        coverage_metrics = self.compute_coverage(smiles_list)
        metrics["coverage"] = coverage_metrics["coverage"]
        metrics["coverage_details"] = coverage_metrics

        print("  Computing diversity...")
        metrics["diversity"] = self.compute_diversity(smiles_list)

        print("  Computing similarity...")
        metrics["similarity"] = self.compute_similarity(smiles_list)

        print("  Computing FCD...")
        metrics["fcd"] = self.compute_fcd(smiles_list)

        return metrics

    def format_comparison_table(self, metrics: Dict[str, any]) -> str:
        header = "| Model | Validity | Coverage | Diversity | Similarity | Distance |"
        separator = "|-------|----------|----------|-----------|------------|----------|"

        rows = [header, separator]

        for model_name, baseline in BACE_BASELINES.items():
            validity_str = f"{baseline['validity']:.4f} ({baseline['validity_filtered']:.4f})"
            row = (
                f"| {model_name} | {validity_str} | {baseline['coverage']} | "
                f"{baseline['diversity']:.4f} | {baseline['similarity']:.4f} | {baseline['fcd']:.4f} |"
            )
            rows.append(row)

        validity_str = f"{metrics['validity']:.4f} ({metrics['validity_filtered']:.4f})"
        row = (
            f"| GRASSY-DiT | {validity_str} | {metrics['coverage']} | "
            f"{metrics['diversity']:.4f} | {metrics['similarity']:.4f} | {metrics['fcd']:.4f} |"
        )
        rows.append(row)

        return "\n".join(rows)

    def run(
        self,
        num_samples: int = 10000,
        batch_size: int = 64,
        output_dir: Optional[str] = None,
        single_index: Optional[int] = None,
        single_samples: int = 1,
    ) -> Dict:
        if output_dir is None:
            output_dir = RESULTS_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        generated_smiles = self.generate_samples(
            num_samples=num_samples,
            batch_size=batch_size,
        )

        metrics = self.compute_all_metrics(generated_smiles)

        single_similarity = None
        if single_index is not None:
            print("\nScoring single-molecule similarity...")
            single_similarity = self.score_single_molecule_similarity(
                index=single_index,
                num_samples=single_samples,
            )
            print(
                "Single molecule similarity:",
                f"Tanimoto mean={single_similarity['tanimoto_mean']:.4f},",
                f"max={single_similarity['tanimoto_max']:.4f},",
                f"FragMetric={single_similarity['frag_similarity']:.4f}",
            )

        samples_path = output_dir / f"bace_generated_{timestamp}.txt"
        with open(samples_path, "w") as f:
            for smi in generated_smiles:
                f.write(smi + "\n")
        print(f"\nSaved generated samples to {samples_path}")

        comparison_table = self.format_comparison_table(metrics)

        print("\n" + "=" * 80)
        print("BACE Benchmark Results")
        print("=" * 80)
        print(f"\nGenerated: {len(generated_smiles)} molecules")
        print(f"Valid: {metrics['n_valid']} ({metrics['validity']:.2%})")
        print("\nMetrics:")
        print(f"  Validity: {metrics['validity']:.4f} ({metrics['validity_filtered']:.4f} filtered)")
        print(f"  Coverage: {metrics['coverage']}")
        print(f"  Diversity: {metrics['diversity']:.4f}")
        print(f"  Similarity: {metrics['similarity']:.4f}")
        print(f"  FCD: {metrics['fcd']:.4f}")
        print("\nComparison with baselines:")
        print(comparison_table)

        results = {
            "config": {
                "num_samples": num_samples,
                "batch_size": batch_size,
                "timestamp": timestamp,
                "single_index": single_index,
                "single_samples": single_samples,
            },
            "metrics": metrics,
            "baselines": BACE_BASELINES,
            "comparison_table": comparison_table,
            "single_similarity": single_similarity,
        }

        results_path = output_dir / f"bace_metrics_{timestamp}.json"

        def convert_to_native(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            return obj

        with open(results_path, "w") as f:
            json.dump(convert_to_native(results), f, indent=2)
        print(f"\nSaved results to {results_path}")

        return results


def main():
    default_config = Path(__file__).parent.parent / "grassy_dit" / "bace_grassy_dit_config.yaml"
    parser = argparse.ArgumentParser(
        description="BACE Benchmark Evaluation for GRASSY-DiT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

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
    parser.add_argument(
        "--smiles-col",
        default="smiles",
        help="Column name for SMILES in csv (default: smiles)",
    )
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
        "--config",
        default=str(default_config),
        help="Path to config yaml (default: grassy_dit/bace_grassy_dit_config.yaml)",
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
    parser.add_argument(
        "--single-index",
        type=int,
        default=None,
        help="Optional test index to compute per-molecule similarity",
    )
    parser.add_argument(
        "--single-samples",
        type=int,
        default=1,
        help="Number of samples for per-molecule similarity (default: 1)",
    )

    args = parser.parse_args()

    benchmark = BACEBenchmark(
        checkpoint=args.checkpoint,
        test_smiles_path=args.test_smiles,
        test_scattering_path=args.test_scattering,
        splits_path=args.splits,
        device=args.device,
        n_jobs=args.n_jobs,
        config_path=args.config,
        smiles_col=args.smiles_col,
    )

    benchmark.run(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        single_index=args.single_index,
        single_samples=args.single_samples,
    )


if __name__ == "__main__":
    main()
