"""
Main entry point for the GRASSY-Net end-to-end pipeline.

Orchestrates all stages from raw data to evaluation reports.

Usage:
    # Run full pipeline
    python -m pipeline.run_pipeline --input data.smi
    
    # Use existing checkpoints
    python -m pipeline.run_pipeline --input data.smi \\
        --dit-checkpoint model.pt --dit-epochs 0
    
    # Control split ratios
    python -m pipeline.run_pipeline --input data.smi \\
        --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15 --split-seed 42
"""

import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.pipeline_config import (
    PipelineConfig,
    load_config,
    save_config,
    merge_cli_args,
)
from pipeline.checkpoint_manager import CheckpointManager
from pipeline.report_generator import ReportGenerator
from pipeline.stages import (
    run_data_prep,
    run_scattering,
    run_splitting,
    run_train_grassy,
    run_train_dit,
    run_evaluate,
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser with all CLI options."""
    parser = argparse.ArgumentParser(
        description='GRASSY-Net End-to-End Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline from scratch
    python -m pipeline.run_pipeline --input datasets/ZINC_tranches/FBAB/FBAB.smi
    
    # Use existing DiT checkpoint, skip training
    python -m pipeline.run_pipeline --input data.smi \\
        --dit-checkpoint checkpoints/model.pt --dit-epochs 0
    
    # Custom split ratios
    python -m pipeline.run_pipeline --input data.smi \\
        --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
        """,
    )
    
    # Required arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input .smi file path (required)',
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory (default: runs/{name}_{timestamp})',
    )
    parser.add_argument(
        '--name',
        type=str,
        default='experiment',
        help='Experiment name (default: experiment)',
    )
    
    # Config file
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='YAML config file (CLI args override config values)',
    )
    
    # Split configuration
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=None,
        help='Training set ratio (default: 0.8)',
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=None,
        help='Validation set ratio (default: 0.1)',
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=None,
        help='Test set ratio (default: 0.1)',
    )
    parser.add_argument(
        '--split-seed',
        type=int,
        default=None,
        help='Random seed for data splitting ONLY (default: 42)',
    )
    
    # GRASSY configuration
    parser.add_argument(
        '--grassy-checkpoint',
        type=str,
        default=None,
        help='Path to existing GRASSY checkpoint',
    )
    parser.add_argument(
        '--grassy-epochs',
        type=int,
        default=None,
        help='GRASSY training epochs (0 to skip, default: 100)',
    )
    
    # DiT configuration
    parser.add_argument(
        '--dit-checkpoint',
        type=str,
        default=None,
        help='Path to existing DiT checkpoint',
    )
    parser.add_argument(
        '--dit-epochs',
        type=int,
        default=None,
        help='DiT training epochs (0 to skip, default: 2000)',
    )
    
    # Noise configuration
    parser.add_argument(
        '--noise-prob',
        type=float,
        default=None,
        help='Probability of adding noise during DiT training (default: 0.2)',
    )
    parser.add_argument(
        '--noise-std',
        type=float,
        default=None,
        help='Gaussian noise standard deviation (default: 0.2)',
    )
    
    # Hardware
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['auto', 'cuda', 'cpu'],
        help='Device to use (default: auto)',
    )
    
    # Overrides
    parser.add_argument(
        '--override',
        type=str,
        nargs='*',
        default=[],
        help='Override config values (e.g., dit.epochs=500)',
    )
    
    # Evaluation
    parser.add_argument(
        '--num-samples',
        type=int,
        default=None,
        help='Number of molecules to generate for evaluation (default: 1000)',
    )
    
    # W&B
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging',
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging',
    )
    
    return parser


def run_pipeline(config: PipelineConfig) -> dict:
    """
    Run the complete pipeline.
    
    Args:
        config: Pipeline configuration
    
    Returns:
        Dictionary with results from all stages
    """
    # Validate config
    config.validate()
    
    # Initialize checkpoint manager
    ckpt_manager = CheckpointManager(config.experiment.output_dir)
    
    # Save config to output directory
    config_path = os.path.join(config.experiment.output_dir, 'pipeline_config.yaml')
    save_config(config, config_path)
    print(f"\nConfig saved to: {config_path}")
    
    results = {}
    
    # =========================================================================
    # Stage 1: Data Preparation
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STAGE 1: DATA PREPARATION")
    print("=" * 70)
    
    stage_start = time.time()
    stage_dir = ckpt_manager.register_stage_start("data_prep")
    
    try:
        # Derive prefix from input filename
        input_name = Path(config.dataset.input_file).stem
        
        data_path, stats_path, metrics = run_data_prep(
            input_file=config.dataset.input_file,
            output_dir=str(stage_dir),
            prefix=input_name,
            max_molecules=config.dataset.max_molecules,
        )
        
        ckpt_manager.register_stage_complete(
            "data_prep",
            checkpoint_path=data_path,
            metrics=metrics,
            duration_seconds=time.time() - stage_start,
        )
        results['data_prep'] = {'data_path': data_path, 'stats_path': stats_path, 'metrics': metrics}
        
    except Exception as e:
        ckpt_manager.register_stage_failed("data_prep", str(e))
        raise
    
    # =========================================================================
    # Stage 2: Scattering Extraction
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STAGE 2: SCATTERING EXTRACTION")
    print("=" * 70)
    
    stage_start = time.time()
    stage_dir = ckpt_manager.register_stage_start("scattering")
    
    try:
        scattering_path, molecules_csv, metrics = run_scattering(
            data_path=results['data_prep']['data_path'],
            stats_path=results['data_prep']['stats_path'],
            output_dir=str(stage_dir),
            J=config.scattering.J,
            num_moments=config.scattering.num_moments,
            device=config.hardware.device,
        )
        
        ckpt_manager.register_stage_complete(
            "scattering",
            checkpoint_path=scattering_path,
            metrics=metrics,
            duration_seconds=time.time() - stage_start,
        )
        results['scattering'] = {
            'scattering_path': scattering_path,
            'molecules_csv': molecules_csv,
            'metrics': metrics
        }
        
    except Exception as e:
        ckpt_manager.register_stage_failed("scattering", str(e))
        raise
    
    # =========================================================================
    # Stage 3: Data Splitting
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STAGE 3: DATA SPLITTING")
    print("=" * 70)
    
    stage_start = time.time()
    stage_dir = ckpt_manager.register_stage_start("splitting")
    
    try:
        train_dir, val_dir, test_dir, metrics = run_splitting(
            scattering_path=results['scattering']['scattering_path'],
            molecules_csv_path=results['scattering']['molecules_csv'],
            output_dir=str(stage_dir),
            train_ratio=config.splitting.train_ratio,
            val_ratio=config.splitting.val_ratio,
            test_ratio=config.splitting.test_ratio,
            seed=config.splitting.seed,  # Deterministic seed
        )
        
        ckpt_manager.register_stage_complete(
            "splitting",
            metrics=metrics,
            duration_seconds=time.time() - stage_start,
        )
        results['splitting'] = {
            'train_dir': train_dir,
            'val_dir': val_dir,
            'test_dir': test_dir,
            'metrics': metrics
        }
        
    except Exception as e:
        ckpt_manager.register_stage_failed("splitting", str(e))
        raise
    
    # =========================================================================
    # Stage 4: GRASSY Training
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STAGE 4: GRASSY TRAINING")
    print("=" * 70)
    
    stage_start = time.time()
    
    # Check if we should skip training
    skip_grassy = ckpt_manager.should_skip_training(
        "train_grassy",
        config.grassy.checkpoint,
        config.grassy.epochs
    )
    
    if skip_grassy:
        print(f"\nSkipping GRASSY training (epochs=0)")
        print(f"Using provided checkpoint: {config.grassy.checkpoint}")
        grassy_checkpoint = ckpt_manager.register_provided_checkpoint(
            "train_grassy",
            config.grassy.checkpoint,
        )
        results['train_grassy'] = {
            'checkpoint': grassy_checkpoint,
            'metrics': {'status': 'skipped', 'checkpoint_source': 'provided'}
        }
    else:
        stage_dir = ckpt_manager.register_stage_start("train_grassy")
        
        try:
            grassy_checkpoint, metrics = run_train_grassy(
                data_path=results['data_prep']['data_path'],
                stats_path=results['data_prep']['stats_path'],
                scattering_path=results['scattering']['scattering_path'],
                output_dir=str(stage_dir),
                checkpoint=config.grassy.checkpoint,
                epochs=config.grassy.epochs,
                bottle_dim=config.grassy.bottle_dim,
                hidden_dim=config.grassy.hidden_dim,
                batch_size=config.grassy.batch_size,
                learning_rate=config.grassy.learning_rate,
                alpha=config.grassy.alpha,
                seed=config.splitting.seed,
                device=config.hardware.device,
                wandb_enabled=config.logging.wandb.enabled,
                wandb_project=config.logging.wandb.project,
                wandb_entity=config.logging.wandb.entity,
            )
            
            ckpt_manager.register_stage_complete(
                "train_grassy",
                checkpoint_path=grassy_checkpoint,
                metrics=metrics,
                duration_seconds=time.time() - stage_start,
            )
            results['train_grassy'] = {'checkpoint': grassy_checkpoint, 'metrics': metrics}
            
        except Exception as e:
            ckpt_manager.register_stage_failed("train_grassy", str(e))
            raise
    
    # =========================================================================
    # Stage 5: DiT Training
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STAGE 5: DiT TRAINING")
    print("=" * 70)
    
    stage_start = time.time()
    
    # Check if we should skip training
    skip_dit = ckpt_manager.should_skip_training(
        "train_dit",
        config.dit.checkpoint,
        config.dit.epochs
    )
    
    if skip_dit:
        print(f"\nSkipping DiT training (epochs=0)")
        print(f"Using provided checkpoint: {config.dit.checkpoint}")
        dit_checkpoint = ckpt_manager.register_provided_checkpoint(
            "train_dit",
            config.dit.checkpoint,
        )
        results['train_dit'] = {
            'checkpoint': dit_checkpoint,
            'metrics': {'status': 'skipped', 'checkpoint_source': 'provided'}
        }
    else:
        stage_dir = ckpt_manager.register_stage_start("train_dit")
        
        try:
            dit_checkpoint, metrics = run_train_dit(
                train_dir=results['splitting']['train_dir'],
                val_dir=results['splitting']['val_dir'],
                output_dir=str(stage_dir),
                checkpoint=config.dit.checkpoint,
                epochs=config.dit.epochs,
                hidden_size=config.dit.hidden_size,
                num_layer=config.dit.num_layer,
                num_head=config.dit.num_head,
                batch_size=config.dit.batch_size,
                learning_rate=config.dit.learning_rate,
                noise_prob=config.dit.noise_prob,
                noise_std=config.dit.noise_std,
                noise_lower=config.dit.noise_lower,
                noise_upper=config.dit.noise_upper,
                J=config.scattering.J,
                num_moments=config.scattering.num_moments,
                device=config.hardware.device,
                wandb_enabled=config.logging.wandb.enabled,
                wandb_project=config.logging.wandb.project,
                wandb_entity=config.logging.wandb.entity,
            )
            
            ckpt_manager.register_stage_complete(
                "train_dit",
                checkpoint_path=dit_checkpoint,
                metrics=metrics,
                duration_seconds=time.time() - stage_start,
            )
            results['train_dit'] = {'checkpoint': dit_checkpoint, 'metrics': metrics}
            
        except Exception as e:
            ckpt_manager.register_stage_failed("train_dit", str(e))
            raise
    
    # =========================================================================
    # Stage 6: Evaluation
    # =========================================================================
    print("\n" + "=" * 70)
    print(" STAGE 6: EVALUATION")
    print("=" * 70)
    
    stage_start = time.time()
    stage_dir = ckpt_manager.register_stage_start("evaluate")
    
    # Find DiT config (either from training output or alongside checkpoint)
    dit_config_path = None
    if not skip_dit:
        dit_config_path = os.path.join(
            os.path.dirname(results['train_dit']['checkpoint']),
            'dit_config.yaml'
        )
    else:
        # Look for config alongside provided checkpoint
        ckpt_dir = os.path.dirname(config.dit.checkpoint)
        for fname in os.listdir(ckpt_dir):
            if fname.endswith('.yaml') or fname.endswith('.yml'):
                dit_config_path = os.path.join(ckpt_dir, fname)
                break
    
    if dit_config_path is None or not os.path.exists(dit_config_path):
        # Create minimal config for evaluation
        dit_config_path = os.path.join(str(stage_dir), 'dit_config.yaml')
        import yaml
        with open(dit_config_path, 'w') as f:
            yaml.dump({
                'scattering': {
                    'J': config.scattering.J,
                    'num_moments': config.scattering.num_moments,
                },
                'model': {
                    'hidden_size': config.dit.hidden_size,
                    'num_layer': config.dit.num_layer,
                    'num_head': config.dit.num_head,
                },
            }, f)
    
    try:
        report_path, metrics = run_evaluate(
            test_dir=results['splitting']['test_dir'],
            train_dir=results['splitting']['train_dir'],
            dit_checkpoint=results['train_dit']['checkpoint'],
            dit_config=dit_config_path,
            output_dir=str(stage_dir),
            grassy_checkpoint=results['train_grassy']['checkpoint'],
            num_samples=config.evaluation.num_samples,
            batch_size=config.evaluation.batch_size,
            guide_scale=config.evaluation.guide_scale,
            device=config.hardware.device,
        )
        
        ckpt_manager.register_stage_complete(
            "evaluate",
            checkpoint_path=report_path,
            metrics=metrics,
            duration_seconds=time.time() - stage_start,
        )
        results['evaluate'] = {'report_path': report_path, 'metrics': metrics}
        
    except Exception as e:
        ckpt_manager.register_stage_failed("evaluate", str(e))
        raise
    
    # =========================================================================
    # Generate Final Report
    # =========================================================================
    print("\n" + "=" * 70)
    print(" GENERATING FINAL REPORT")
    print("=" * 70)
    
    report_gen = ReportGenerator(config.experiment.output_dir)
    final_report = report_gen.generate_report(ckpt_manager, config, results)
    
    # Finalize
    ckpt_manager.finalize("completed")
    
    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {config.experiment.output_dir}")
    print(f"Final report: {final_report}")
    
    return results


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config:
        config = load_config(args.config)
    else:
        config = PipelineConfig()
    
    # Set experiment name
    if hasattr(args, 'name') and args.name:
        config.experiment.name = args.name
    
    # Merge CLI arguments (they override config file)
    config = merge_cli_args(config, args)
    
    # Handle W&B flags
    if args.wandb:
        config.logging.wandb.enabled = True
    if args.no_wandb:
        config.logging.wandb.enabled = False
    
    # Handle num_samples
    if args.num_samples:
        config.evaluation.num_samples = args.num_samples
    
    # Run pipeline
    try:
        results = run_pipeline(config)
        return 0
    except Exception as e:
        print(f"\n\nPIPELINE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
