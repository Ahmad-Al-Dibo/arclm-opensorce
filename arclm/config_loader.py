"""
Configuration File Loading

Supports loading configurations from YAML and JSON files.

Example YAML config:
    # config.yaml
    model:
      embed_dim: 128
      num_blocks: 4
      block_size: 512
      num_heads: 4
    
    training:
      learning_rate: 0.001
      batch_size: 16
      epochs: 5
    
    data:
      tokenizer_type: "word"
      vocab_size: 10000

Example JSON config:
    {
      "model": {
        "embed_dim": 128,
        "num_blocks": 4
      },
      "training": {
        "learning_rate": 0.001
      }
    }
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from .config import Config


def load_config_yaml(filepath: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        filepath: Path to YAML config file
    
    Returns:
        Config object
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If YAML library not installed or file invalid
    
    Example:
        config = load_config_yaml("config.yaml")
    """
    if not YAML_AVAILABLE:
        raise ValueError(
            "PyYAML not installed. Install with: pip install pyyaml"
        )
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    try:
        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML file: {e}")
    
    if not isinstance(config_dict, dict):
        raise ValueError("Config must be a YAML dictionary")
    
    # Flatten nested structure
    flat_config = _flatten_config(config_dict)
    
    return Config(**flat_config)


def load_config_json(filepath: str) -> Config:
    """
    Load configuration from JSON file.
    
    Args:
        filepath: Path to JSON config file
    
    Returns:
        Config object
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON is invalid
    
    Example:
        config = load_config_json("config.json")
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")
    
    try:
        with open(filepath, "r") as f:
            config_dict = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}")
    
    if not isinstance(config_dict, dict):
        raise ValueError("Config must be a JSON object/dictionary")
    
    # Flatten nested structure
    flat_config = _flatten_config(config_dict)
    
    return Config(**flat_config)


def load_config(filepath: str) -> Config:
    """
    Load configuration from file (auto-detect format).
    
    Automatically detects format based on file extension.
    
    Args:
        filepath: Path to config file (.yaml, .yml, or .json)
    
    Returns:
        Config object
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format not recognized or invalid
    
    Example:
        config = load_config("config.yaml")  # Auto-detects YAML
        config = load_config("settings.json")  # Auto-detects JSON
    """
    filepath = Path(filepath)
    
    if filepath.suffix in [".yaml", ".yml"]:
        return load_config_yaml(str(filepath))
    elif filepath.suffix == ".json":
        return load_config_json(str(filepath))
    else:
        raise ValueError(
            f"Unknown config format: {filepath.suffix}. "
            "Supported: .yaml, .yml, .json"
        )


def save_config_yaml(config: Config, filepath: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object to save
        filepath: Path to save YAML file
    
    Example:
        save_config_yaml(config, "config.yaml")
    """
    if not YAML_AVAILABLE:
        raise ValueError(
            "PyYAML not installed. Install with: pip install pyyaml"
        )
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    try:
        with open(filepath, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise ValueError(f"Error saving YAML file: {e}")


def save_config_json(config: Config, filepath: str) -> None:
    """
    Save configuration to JSON file.
    
    Args:
        config: Config object to save
        filepath: Path to save JSON file
    
    Example:
        save_config_json(config, "config.json")
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    try:
        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2)
    except Exception as e:
        raise ValueError(f"Error saving JSON file: {e}")


def save_config(config: Config, filepath: str) -> None:
    """
    Save configuration to file (auto-detect format).
    
    Args:
        config: Config object to save
        filepath: Path to save (.yaml, .yml, or .json)
    
    Example:
        save_config(config, "config.yaml")
        save_config(config, "config.json")
    """
    filepath = Path(filepath)
    
    if filepath.suffix in [".yaml", ".yml"]:
        save_config_yaml(config, str(filepath))
    elif filepath.suffix == ".json":
        save_config_json(config, str(filepath))
    else:
        raise ValueError(
            f"Unknown config format: {filepath.suffix}. "
            "Supported: .yaml, .yml, .json"
        )


def _flatten_config(config_dict: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    """
    Flatten nested configuration dictionary.
    
    Converts nested dict like {'model': {'embed_dim': 128}} 
    to flat dict like {'embed_dim': 128}
    
    Args:
        config_dict: Nested configuration dictionary
        prefix: Internal prefix for recursion
    
    Returns:
        Flattened configuration dictionary
    
    Example:
        nested = {
            'model': {'embed_dim': 128, 'num_blocks': 4},
            'training': {'learning_rate': 0.001}
        }
        flat = _flatten_config(nested)
        # Result: {'embed_dim': 128, 'num_blocks': 4, 'learning_rate': 0.001}
    """
    flat = {}
    
    for key, value in config_dict.items():
        if isinstance(value, dict):
            # Recursively flatten nested dicts
            nested_flat = _flatten_config(value, prefix=key)
            flat.update(nested_flat)
        else:
            # Keep scalar values
            flat[key] = value
    
    return flat


def create_example_configs():
    """
    Create example config files in configs/ directory.
    
    Useful for users to understand config structure.
    """
    from .config import Config
    
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Small model config
    small_config = Config(
        embed_dim=64,
        num_blocks=2,
        block_size=256,
        num_heads=2,
        learning_rate=1e-3,
        batch_size=8,
    )
    save_config_yaml(small_config, str(configs_dir / "small.yaml"))
    save_config_json(small_config, str(configs_dir / "small.json"))
    
    # Medium model config
    medium_config = Config(
        embed_dim=128,
        num_blocks=4,
        block_size=512,
        num_heads=4,
        learning_rate=5e-4,
        batch_size=16,
    )
    save_config_yaml(medium_config, str(configs_dir / "medium.yaml"))
    save_config_json(medium_config, str(configs_dir / "medium.json"))
    
    # Large model config
    large_config = Config(
        embed_dim=256,
        num_blocks=8,
        block_size=1024,
        num_heads=8,
        learning_rate=1e-4,
        batch_size=32,
    )
    save_config_yaml(large_config, str(configs_dir / "large.yaml"))
    save_config_json(large_config, str(configs_dir / "large.json"))
    
    print(f"[OK] Example configs created in {configs_dir}/")
