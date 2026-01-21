
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
        beta=config['training']['beta'],
        n_epochs=config['training']['n_epochs'],
        len_epoch=len_epoch,
        num_properties=num_properties,
        n_gpus=config['hardware']['n_gpus'],
    )
    return hparams

def get_grassy_flags(grassy_version: str) -> tuple:
    """Get kl_div and reg flags based on GRASSY version."""
    version_map = {
        'AE+REG': (False, True),
        'VAE+REG': (True, True),
        'AE': (False, False),
        'VAE': (True, False),
    }
    if grassy_version not in version_map:
        raise ValueError(f"Unknown GRASSY version: {grassy_version}. Options: {list(version_map.keys())}")
    return version_map[grassy_version]