"""
Model configuration management.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider: str
    base_url: str
    description: str
    max_tokens: int = 4000
    temperature: float = 0.3
    api_key: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "provider": self.provider,
            "base_url": self.base_url,
            "description": self.description,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "api_key": self.api_key
        }


def load_model_configs(config_path: Optional[Path] = None) -> Dict[str, ModelConfig]:
    """
    Load model configurations from YAML file.
    
    Args:
        config_path: Path to models configuration file
        
    Returns:
        Dictionary mapping model names to ModelConfig objects
    """
    if config_path is None:
        config_path = Path(__file__).parent / "models.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Model config file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        model_configs = {}
        
        for provider, models in config_data.items():
            for model_name, model_data in models.items():
                config = ModelConfig(
                    name=model_data["name"],
                    provider=model_data["provider"],
                    base_url=model_data["base_url"],
                    description=model_data["description"],
                    max_tokens=model_data.get("max_tokens", 4000),
                    temperature=model_data.get("temperature", 0.3)
                )
                model_configs[model_name] = config
        
        return model_configs
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model configs from {config_path}: {e}")


def get_model_config(model_name: str, config_path: Optional[Path] = None) -> Optional[ModelConfig]:
    """
    Get configuration for a specific model.
    
    Args:
        model_name: Name of the model
        config_path: Path to models configuration file
        
    Returns:
        ModelConfig object or None if not found
    """
    configs = load_model_configs(config_path)
    return configs.get(model_name)


def list_available_models(config_path: Optional[Path] = None) -> Dict[str, list]:
    """
    List all available models grouped by provider.
    
    Args:
        config_path: Path to models configuration file
        
    Returns:
        Dictionary mapping provider names to lists of model names
    """
    configs = load_model_configs(config_path)
    
    providers = {}
    for config in configs.values():
        if config.provider not in providers:
            providers[config.provider] = []
        providers[config.provider].append(config.name)
    
    return providers
