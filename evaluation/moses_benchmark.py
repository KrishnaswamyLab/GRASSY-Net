"""
MOSES Benchmark Evaluation Script for GRASSY-DiT.

Evaluates GRASSY-DiT molecular generation against MOSES baselines
and DiGress using standard MOSES metrics.

Four Evaluation Modes:
- unconditional: DiT generates randomly (no conditioning)
- direct: molecule → scattering → DiT
- full-pipeline: molecule → scattering → GRASSY VAE → moments → DiT
- latent-sample: random latent z → GRASSY decode → moments → DiT

Usage:
    python -m evaluation.moses_benchmark \\
        --dit-checkpoint grassy_dit_checkpoint.pt \\
        --mode direct \\
        --num-samples 10000
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "moses"))

from evaluation.utils import (
    load_dit_model,
    load_grassy_vae,
    load_scattering_model,
    get_conditioning,
    get_atom_types_from_model,
    format_metrics_table,
)
from evaluation.load_baselines import (
    load_baseline_samples,
    load_baseline_metrics,
    load_all_baseline_metrics,
    load_moses_train_set,
    load_moses_test_set,
    check_digress_available,
    generate_digress_samples,
    print_available_baselines,
    MOSES_BASELINES,
    ensure_moses_stats,
)

# Output directory
RESULTS_DIR = Path(__file__).parent / "results"


class MOSESBenchmark:
    """
    MOSES benchmark evaluation for GRASSY-DiT.

    Supports four generation modes and comparison with baselines.
    """

    MODES = ["unconditional", "direct", "full-pipeline", "latent-sample"]

    def __init__(
        self,
        dit_checkpoint: str,
        grassy_checkpoint: Optional[str] = None,
        scattering_checkpoint: Optional[str] = None,
        device: str = "cuda",
        n_jobs: int = 4,
    ):
        """
        Initialize the benchmark.

        Args:
            dit_checkpoint: Path to trained GRASSY-DiT checkpoint
            grassy_checkpoint: Path to GRASSY VAE (for full-pipeline/latent-sample)
            scattering_checkpoint: Path to learnable scattering (None = fixed)
            device: Computation device
            n_jobs: Number of workers for metric computation
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.n_jobs = n_jobs

        # Load models
        print(f"Loading DiT model from {dit_checkpoint}...")
        self.dit_model = load_dit_model(dit_checkpoint, self.device)

        self.grassy_vae = None
        if grassy_checkpoint is not None:
            print(f"Loading GRASSY VAE from {grassy_checkpoint}...")
            self.grassy_vae = load_grassy_vae(grassy_checkpoint, self.device)

        # Get atom types from DiT model
        self.atom_types = get_atom_types_from_model(self.dit_model)
        print(f"Atom types: {self.atom_types}")

        # Load scattering model
        print("Loading scattering model...")
        self.scattering_model = load_scattering_model(
            scattering_checkpoint,
            in_channels=len(self.atom_types),
            device=self.device,
        )

        # Load MOSES data
        print("Loading MOSES datasets...")
        self.train_smiles = load_moses_train_set()
        self.test_smiles = load_moses_test_set(scaffold=False)
        print(f"  Train: {len(self.train_smiles)} molecules")
        print(f"  Test: {len(self.test_smiles)} molecules")

    def generate_samples(
        self,
        mode: str,
        num_samples: int = 10000,
        source: str = "train",
        batch_size: int = 64,
    ) -> List[str]:
        """
        Generate molecules using GRASSY-DiT.

        Args:
            mode: Generation mode (unconditional, direct, full-pipeline, latent-sample)
            num_samples: Number of molecules to generate
            source: Source for conditioning ('train' or 'test')
            batch_size: Generation batch size

        Returns:
            List of generated SMILES strings
        """
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode: {mode}. Expected one of {self.MODES}")

        if mode in ["full-pipeline", "latent-sample"] and self.grassy_vae is None:
            raise ValueError(f"GRASSY VAE required for '{mode}' mode")

        # Get source molecules for conditioning
        source_smiles = self.train_smiles if source == "train" else self.test_smiles

        print(f"\nGenerating {num_samples} molecules in '{mode}' mode...")

        # Get conditioning vectors
        conditioning = get_conditioning(
            mode=mode,
            source_smiles=source_smiles,
            scattering_model=self.scattering_model,
            grassy_vae=self.grassy_vae,
            num_samples=num_samples,
            device=self.device,
            atom_types=self.atom_types,
        )

        # Generate molecules
        all_smiles = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Generating"):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, num_samples)
            current_batch_size = batch_end - batch_start

            if conditioning is not None:
                batch_conditioning = conditioning[batch_start:batch_end].to(self.device)
                smiles_batch = self.dit_model.generate(
                    scattering=batch_conditioning,
                    num_nodes=None,  # Sample from training distribution
                    batch_size=current_batch_size,
                )
            else:
                # Unconditional generation - dummy scattering + CFG null path
                tokenizer = self.dit_model.model.denoiser.scatter_tokenizer
                scattering_dim = tokenizer.num_atom_types * tokenizer.num_levels * tokenizer.num_moments
                dummy_scattering = torch.ones(current_batch_size, scattering_dim, device=self.device)
                original_guide_scale = self.dit_model.guide_scale
                self.dit_model.guide_scale = 0.0
                smiles_batch = self.dit_model.generate(
                    scattering=dummy_scattering,
                    num_nodes=None,
                    batch_size=current_batch_size,
                )
                self.dit_model.guide_scale = original_guide_scale

            # Filter None values
            valid_smiles = [s for s in smiles_batch if s is not None]
            all_smiles.extend(valid_smiles)

        print(f"Generated {len(all_smiles)}/{num_samples} valid molecules")
        return all_smiles

    def compute_metrics(
        self,
        generated_smiles: List[str],
        k: List[int] = [1000, 10000],
    ) -> Dict[str, float]:
        """
        Compute MOSES metrics for generated molecules.

        Args:
            generated_smiles: List of generated SMILES
            k: Values for unique@k metric

        Returns:
            Dictionary of metric names to values
        """
        from moses.metrics import get_all_metrics

        print(f"\nComputing MOSES metrics for {len(generated_smiles)} molecules...")
        ensure_moses_stats()  

        metrics = get_all_metrics(
            gen=generated_smiles,
            k=k,
            n_jobs=self.n_jobs,
            device=self.device,
            train=self.train_smiles,
        )

        return metrics

    def run(
        self,
        mode: str = "direct",
        source: str = "train",
        num_samples: int = 10000,
        batch_size: int = 64,
        compare_baselines: bool = True,
        compare_digress: bool = False,
        output_dir: Optional[str] = None,
    ) -> Dict:
        """
        Run the full benchmark evaluation.

        Args:
            mode: Generation mode
            source: Source for conditioning
            num_samples: Number of molecules to generate
            batch_size: Generation batch size
            compare_baselines: Compare against MOSES baselines
            compare_digress: Compare against DiGress
            output_dir: Output directory (default: evaluation/results/)

        Returns:
            Dictionary containing all results
        """
        if output_dir is None:
            output_dir = RESULTS_DIR
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        results = {
            "config": {
                "mode": mode,
                "source": source,
                "num_samples": num_samples,
                "batch_size": batch_size,
                "timestamp": timestamp,
            },
            "grassy_dit": {},
            "baselines": {},
        }

        # Generate and evaluate GRASSY-DiT
        generated_smiles = self.generate_samples(
            mode=mode,
            num_samples=num_samples,
            source=source,
            batch_size=batch_size,
        )

        results["grassy_dit"] = self.compute_metrics(generated_smiles)

        # Save generated samples
        samples_path = output_dir / f"grassy_dit_samples_{mode}_{timestamp}.txt"
        with open(samples_path, "w") as f:
            for smi in generated_smiles:
                f.write(smi + "\n")
        print(f"Saved generated samples to {samples_path}")

        # Load baseline metrics for comparison
        if compare_baselines:
            print("\nLoading baseline metrics...")
            results["baselines"] = load_all_baseline_metrics(run_id=1)

        # Compare with DiGress
        if compare_digress:
            if check_digress_available():
                print("\nGenerating DiGress samples...")
                digress_smiles = generate_digress_samples(
                    num_samples=num_samples,
                    batch_size=batch_size,
                    device=self.device,
                )
                results["baselines"]["digress"] = self.compute_metrics(digress_smiles)
            else:
                print("DiGress checkpoint not found, skipping comparison")

        # Create comparison table
        all_metrics = {"GRASSY-DiT": results["grassy_dit"]}
        all_metrics.update(results["baselines"])

        model_order = ["GRASSY-DiT"] + MOSES_BASELINES
        if "digress" in results["baselines"]:
            model_order.append("digress")

        results["comparison_table"] = format_metrics_table(
            all_metrics, model_names=model_order
        )

        # Print results
        print("\n" + "=" * 60)
        print("MOSES Benchmark Results")
        print("=" * 60)
        print(f"\nMode: {mode}")
        print(f"Source: {source}")
        print(f"Generated: {len(generated_smiles)} molecules")
        print("\nGRASSY-DiT Metrics:")
        for key, value in sorted(results["grassy_dit"].items()):
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print("\n" + results["comparison_table"])

        # Save results
        results_path = output_dir / f"moses_benchmark_{mode}_{timestamp}.json"
        
        # Convert numpy types to native Python types for JSON serialization
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

        results_native = convert_to_native(results)
        
        with open(results_path, "w") as f:
            json.dump(results_native, f, indent=2)
        print(f"\nSaved results to {results_path}")

        return results


def run_benchmark(**kwargs) -> Dict:
    """
    Convenience function to run MOSES benchmark.

    See MOSESBenchmark.run() for available arguments.
    """
    benchmark = MOSESBenchmark(
        dit_checkpoint=kwargs.pop("dit_checkpoint"),
        grassy_checkpoint=kwargs.pop("grassy_checkpoint", None),
        scattering_checkpoint=kwargs.pop("scattering_checkpoint", None),
        device=kwargs.pop("device", "cuda"),
        n_jobs=kwargs.pop("n_jobs", 4),
    )

    return benchmark.run(**kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="MOSES Benchmark Evaluation for GRASSY-DiT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mode 0: Unconditional generation
  python -m evaluation.moses_benchmark \\
      --dit-checkpoint grassy_dit_checkpoint.pt \\
      --mode unconditional \\
      --num-samples 10000

  # Mode 1: Direct scattering conditioning (default)
  python -m evaluation.moses_benchmark \\
      --dit-checkpoint grassy_dit_checkpoint.pt \\
      --mode direct \\
      --source train

  # Mode 2: Full pipeline with GRASSY VAE round-trip
  python -m evaluation.moses_benchmark \\
      --dit-checkpoint grassy_dit_checkpoint.pt \\
      --grassy-checkpoint grassy_vae.pt \\
      --mode full-pipeline

  # Mode 3: Sample from GRASSY latent space
  python -m evaluation.moses_benchmark \\
      --dit-checkpoint grassy_dit_checkpoint.pt \\
      --grassy-checkpoint grassy_vae.pt \\
      --mode latent-sample

  # With DiGress comparison
  python -m evaluation.moses_benchmark \\
      --dit-checkpoint grassy_dit_checkpoint.pt \\
      --compare-digress
        """,
    )

    # Required arguments
    parser.add_argument(
        "--dit-checkpoint",
        required=True,
        help="Path to trained GRASSY-DiT model checkpoint",
    )

    # Optional model checkpoints
    parser.add_argument(
        "--grassy-checkpoint",
        default=None,
        help="Path to GRASSY VAE (required for full-pipeline and latent-sample modes)",
    )
    parser.add_argument(
        "--scattering-checkpoint",
        default=None,
        help="Path to learnable scattering model (uses fixed GraphScatteringTransform if not provided)",
    )

    # Generation settings
    parser.add_argument(
        "--mode",
        choices=MOSESBenchmark.MODES,
        default="direct",
        help="Generation mode: unconditional, direct, full-pipeline, latent-sample",
    )
    parser.add_argument(
        "--source",
        choices=["train", "test"],
        default="train",
        help="Source molecules for conditioning (train or test set)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of molecules to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Generation batch size",
    )

    # Comparison settings
    parser.add_argument(
        "--compare-baselines",
        action="store_true",
        default=True,
        help="Compare against MOSES pre-generated baselines",
    )
    parser.add_argument(
        "--no-compare-baselines",
        action="store_false",
        dest="compare_baselines",
        help="Skip baseline comparison",
    )
    parser.add_argument(
        "--compare-digress",
        action="store_true",
        default=False,
        help="Compare against DiGress (requires checkpoint in evaluation/checkpoints/digress/)",
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: evaluation/results/)",
    )

    # Computation settings
    parser.add_argument(
        "--device",
        default="cuda",
        help="Computation device (cuda or cpu)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Number of workers for metric computation",
    )

    # Utility commands
    parser.add_argument(
        "--list-baselines",
        action="store_true",
        help="List available baselines and exit",
    )

    args = parser.parse_args()

    # Handle utility commands
    if args.list_baselines:
        print_available_baselines()
        return

    # Validate arguments
    if args.mode in ["full-pipeline", "latent-sample"] and args.grassy_checkpoint is None:
        parser.error(f"--grassy-checkpoint is required for '{args.mode}' mode")

    # Run benchmark
    benchmark = MOSESBenchmark(
        dit_checkpoint=args.dit_checkpoint,
        grassy_checkpoint=args.grassy_checkpoint,
        scattering_checkpoint=args.scattering_checkpoint,
        device=args.device,
        n_jobs=args.n_jobs,
    )

    results = benchmark.run(
        mode=args.mode,
        source=args.source,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        compare_baselines=args.compare_baselines,
        compare_digress=args.compare_digress,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
