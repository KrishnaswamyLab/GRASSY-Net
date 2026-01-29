"""
Pipeline configuration schema and utilities.

Handles loading, validation, and merging of config from YAML files and CLI arguments.
"""

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from pathlib import Path
import yaml
from datetime import datetime


@dataclass
class ExperimentConfig:
    """Experiment metadata."""
    name: str = "experiment"
    output_dir: Optional[str] = None  # Auto-generated if None
    
    def __post_init__(self):
        if self.output_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.output_dir = f"runs/{self.name}_{timestamp}"


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    input_file: str = ""  # Required: path to .smi file
    max_molecules: Optional[int] = None  # Limit number of molecules (None = all)


@dataclass
class SplittingConfig:
    """Data splitting configuration."""
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42  # ONLY used for splitting - ensures reproducible partitions
    
    def __post_init__(self):
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class ScatteringConfig:
    """Scattering transform configuration."""
    J: int = 4  # Number of wavelet scales
    num_moments: int = 4  # Statistical moments (mean, var, skew, kurtosis)


@dataclass
class GRASSYConfig:
    """GRASSY autoencoder configuration."""
    checkpoint: Optional[str] = None  # Path to existing checkpoint
    epochs: int = 100  # Additional epochs to train (0 = use checkpoint as-is)
    bottle_dim: int = 32
    hidden_dim: int = 128
    batch_size: int = 64
    learning_rate: float = 0.001
    alpha: float = 0.01  # Regression loss weight


@dataclass
class DiTConfig:
    """DiT (Diffusion Transformer) configuration."""
    checkpoint: Optional[str] = None  # Path to existing checkpoint
    epochs: int = 2000  # Additional epochs to train (0 = use checkpoint as-is)
    hidden_size: int = 1024
    num_layer: int = 6
    num_head: int = 16
    batch_size: int = 64
    learning_rate: float = 2e-4
    # Moment noise augmentation
    noise_prob: float = 0.2  # Probability of adding noise to a sample
    noise_std: float = 0.2  # Gaussian noise standard deviation
    noise_lower: float = 0.25  # Min fraction of moments to corrupt
    noise_upper: float = 1.0  # Max fraction of moments to corrupt


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    num_samples: int = 1000  # Number of molecules to generate
    batch_size: int = 64
    recon_samples: int = 50  # Molecules for reconstruction test
    recon_attempts: int = 10  # Attempts per molecule
    guide_scale: float = 2.0  # Classifier-free guidance scale


@dataclass
class HardwareConfig:
    """Hardware configuration."""
    device: str = "auto"  # "auto", "cuda", "cpu"
    num_workers: int = 4


@dataclass
class WandbConfig:
    """Weights & Biases logging configuration."""
    enabled: bool = True
    project: str = "GRASSY-Pipeline"
    entity: str = "grassy"


@dataclass
class LoggingConfig:
    """Logging configuration."""
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass
class PipelineConfig:
    """
    Complete pipeline configuration.
    
    All settings for running the end-to-end GRASSY-Net pipeline.
    """
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    splitting: SplittingConfig = field(default_factory=SplittingConfig)
    scattering: ScatteringConfig = field(default_factory=ScatteringConfig)
    grassy: GRASSYConfig = field(default_factory=GRASSYConfig)
    dit: DiTConfig = field(default_factory=DiTConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def validate(self) -> None:
        """Validate configuration and raise errors for invalid settings."""
        # Check input file exists
        if not self.dataset.input_file:
            raise ValueError("dataset.input_file is required")
        if not os.path.exists(self.dataset.input_file):
            raise FileNotFoundError(f"Input file not found: {self.dataset.input_file}")
        
        # Check checkpoint logic
        if self.grassy.checkpoint is None and self.grassy.epochs == 0:
            raise ValueError("Cannot skip GRASSY training (epochs=0) without providing a checkpoint")
        if self.dit.checkpoint is None and self.dit.epochs == 0:
            raise ValueError("Cannot skip DiT training (epochs=0) without providing a checkpoint")
        
        # Check provided checkpoints exist
        if self.grassy.checkpoint and not os.path.exists(self.grassy.checkpoint):
            raise FileNotFoundError(f"GRASSY checkpoint not found: {self.grassy.checkpoint}")
        if self.dit.checkpoint and not os.path.exists(self.dit.checkpoint):
            raise FileNotFoundError(f"DiT checkpoint not found: {self.dit.checkpoint}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


def load_config(config_path: Optional[str] = None) -> PipelineConfig:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, returns default config.
    
    Returns:
        PipelineConfig instance
    """
    if config_path is None:
        return PipelineConfig()
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return _dict_to_config(config_dict)


def _dict_to_config(d: Dict[str, Any]) -> PipelineConfig:
    """Convert nested dict to PipelineConfig."""
    config = PipelineConfig()
    
    if 'experiment' in d:
        config.experiment = ExperimentConfig(**d['experiment'])
    if 'dataset' in d:
        config.dataset = DatasetConfig(**d['dataset'])
    if 'splitting' in d:
        config.splitting = SplittingConfig(**d['splitting'])
    if 'scattering' in d:
        config.scattering = ScatteringConfig(**d['scattering'])
    if 'grassy' in d:
        config.grassy = GRASSYConfig(**d['grassy'])
    if 'dit' in d:
        config.dit = DiTConfig(**d['dit'])
    if 'evaluation' in d:
        config.evaluation = EvaluationConfig(**d['evaluation'])
    if 'hardware' in d:
        config.hardware = HardwareConfig(**d['hardware'])
    if 'logging' in d:
        logging_dict = d['logging']
        if 'wandb' in logging_dict:
            config.logging = LoggingConfig(wandb=WandbConfig(**logging_dict['wandb']))
    
    return config


def save_config(config: PipelineConfig, path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def apply_cli_overrides(config: PipelineConfig, overrides: Dict[str, Any]) -> PipelineConfig:
    """
    Apply CLI argument overrides to config.
    
    Args:
        config: Base configuration
        overrides: Dict of overrides like {'dit.epochs': 0, 'grassy.checkpoint': 'path'}
    
    Returns:
        Modified config
    """
    for key, value in overrides.items():
        parts = key.split('.')
        if len(parts) == 2:
            section, attr = parts
            if hasattr(config, section):
                section_obj = getattr(config, section)
                if hasattr(section_obj, attr):
                    setattr(section_obj, attr, value)
    return config


def merge_cli_args(config: PipelineConfig, args) -> PipelineConfig:
    """
    Merge argparse namespace into config.
    
    CLI arguments take precedence over config file values.
    """
    # Dataset
    if hasattr(args, 'input') and args.input:
        config.dataset.input_file = args.input
    
    # Output directory
    if hasattr(args, 'output_dir') and args.output_dir:
        config.experiment.output_dir = args.output_dir
    
    # Splitting
    if hasattr(args, 'train_ratio') and args.train_ratio is not None:
        config.splitting.train_ratio = args.train_ratio
    if hasattr(args, 'val_ratio') and args.val_ratio is not None:
        config.splitting.val_ratio = args.val_ratio
    if hasattr(args, 'test_ratio') and args.test_ratio is not None:
        config.splitting.test_ratio = args.test_ratio
    if hasattr(args, 'split_seed') and args.split_seed is not None:
        config.splitting.seed = args.split_seed
    
    # GRASSY
    if hasattr(args, 'grassy_checkpoint') and args.grassy_checkpoint:
        config.grassy.checkpoint = args.grassy_checkpoint
    if hasattr(args, 'grassy_epochs') and args.grassy_epochs is not None:
        config.grassy.epochs = args.grassy_epochs
    
    # DiT
    if hasattr(args, 'dit_checkpoint') and args.dit_checkpoint:
        config.dit.checkpoint = args.dit_checkpoint
    if hasattr(args, 'dit_epochs') and args.dit_epochs is not None:
        config.dit.epochs = args.dit_epochs
    if hasattr(args, 'noise_prob') and args.noise_prob is not None:
        config.dit.noise_prob = args.noise_prob
    if hasattr(args, 'noise_std') and args.noise_std is not None:
        config.dit.noise_std = args.noise_std
    
    # Hardware
    if hasattr(args, 'device') and args.device:
        config.hardware.device = args.device
    
    # Apply --override arguments
    if hasattr(args, 'override') and args.override:
        for override in args.override:
            if '=' in override:
                key, value = override.split('=', 1)
                # Try to parse value as int, float, bool, or keep as string
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        elif value.lower() == 'none':
                            value = None
                config = apply_cli_overrides(config, {key: value})
    
    return config
