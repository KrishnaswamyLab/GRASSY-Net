
from types import SimpleNamespace
import yaml

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def apply_overrides(config: dict, overrides: list) -> dict:
    """Apply command-line overrides to config.

    Example: --override training.n_epochs=50 model.hidden_dim=200
    """
    for override in overrides:
        if '=' not in override:
            raise ValueError(f"Invalid override format: {override}. Use key.subkey=value")

        key_path, value = override.split('=', 1)
        keys = key_path.split('.')

        # Navigate to the parent dict
        d = config
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]

        # Try to parse the value as the appropriate type
        final_key = keys[-1]
        try:
            parsed_value = int(value)
        except ValueError:
            try:
                parsed_value = float(value)
            except ValueError:
                if value.lower() in ('true', 'false'):
                    parsed_value = value.lower() == 'true'
                elif value.lower() == 'null' or value.lower() == 'none':
                    parsed_value = None
                else:
                    parsed_value = value

        d[final_key] = parsed_value
        print(f"Override applied: {key_path} = {parsed_value}")

    return config


def config_to_hparams(config: dict, input_dim: int, num_properties: int, len_epoch: int) -> SimpleNamespace:
    """Convert config dict to hparams namespace for GRASSY model."""
    hparams = SimpleNamespace(
        input_dim=input_dim,
        bottle_dim=config['model']['bottle_dim'],
        hidden_dim=config['model']['hidden_dim'],
        learning_rate=config['training']['learning_rate'],
        alpha=config['training']['alpha'],
        atom_loss_weight=config['training'].get('atom_loss_weight', 1.0),
        num_atom_classes=config['model'].get('num_atom_classes', config['dataset'].get('num_atom_classes', 64)),
        n_epochs=config['training']['n_epochs'],
        len_epoch=len_epoch,
        num_properties=num_properties,
        n_gpus=config['hardware']['n_gpus'],
    )
    return hparams

def calculate_split_sizes(total_size: int, train_pct: float, val_pct: float, test_pct: float) -> tuple:
    """Calculate split sizes from percentages.
    
    Args:
        total_size: Total number of samples in the dataset
        train_pct: Training set percentage (0-100)
        val_pct: Validation set percentage (0-100)
        test_pct: Test set percentage (0-100)
    
    Returns:
        Tuple of (train_size, val_size, test_size)
    """
    # Validate percentages
    total_pct = train_pct + val_pct + test_pct
    if not (99.9 <= total_pct <= 100.1):  
        raise ValueError(f"Split percentages must sum to 100, got {total_pct} "
                        f"(train={train_pct}, val={val_pct}, test={test_pct})")
    
    # Calculate sizes
    train_size = int(total_size * train_pct / 100)
    val_size = int(total_size * val_pct / 100)
    
    # Assign remainder to test set to ensure all samples are used
    test_size = total_size - train_size - val_size
    
    return train_size, val_size, test_size