"""
Application settings and configuration management.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class TranslationConfig(BaseModel):
    """Translation configuration."""
    max_file_size: int = 100 * 1024  # 100KB
    max_concurrent: int = 3
    temperature: float = 0.3
    max_tokens: Optional[int] = 4000
    source_lang: str = "en"
    target_lang: str = "zh"


class FileConfig(BaseModel):
    """File processing configuration."""
    supported_extensions: list[str] = [".md", ".markdown"]
    create_backup: bool = True
    validate_translation: bool = True
    output_suffix: str = "_zh"


@dataclass
class Settings:
    """Application settings."""
    # Model configuration
    model_provider: str = "ollama"
    model_name: str = "qwen2.5:7b"
    
    # API configurations
    ollama_url: str = "http://localhost:11434"
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None
    claude_api_key: Optional[str] = None
    qwen_api_key: Optional[str] = None
    qwen_base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    
    # Translation settings
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    
    # File processing settings
    file_config: FileConfig = field(default_factory=FileConfig)
    
    # Logging settings
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Working directory
    work_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Initialize settings after creation."""
        if self.work_dir is None:
            self.work_dir = Path.cwd()
        
        # Load API keys from environment variables if not set
        if not self.openai_api_key:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.claude_api_key:
            self.claude_api_key = os.getenv("CLAUDE_API_KEY")
        
        if not self.qwen_api_key:
            self.qwen_api_key = os.getenv("QWEN_API_KEY")


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """
    Load settings from configuration file and environment variables.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Settings object
    """
    settings = Settings()
    
    if config_path and config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Update settings with config file data
            if 'model' in config_data:
                model_config = config_data['model']
                settings.model_provider = model_config.get('provider', settings.model_provider)
                settings.model_name = model_config.get('name', settings.model_name)
            
            if 'api' in config_data:
                api_config = config_data['api']
                settings.ollama_url = api_config.get('ollama_url', settings.ollama_url)
                settings.openai_api_key = api_config.get('openai_api_key', settings.openai_api_key)
                settings.openai_base_url = api_config.get('openai_base_url', settings.openai_base_url)
                settings.claude_api_key = api_config.get('claude_api_key', settings.claude_api_key)
                settings.qwen_api_key = api_config.get('qwen_api_key', settings.qwen_api_key)
                settings.qwen_base_url = api_config.get('qwen_base_url', settings.qwen_base_url)
            
            if 'translation' in config_data:
                translation_config = config_data['translation']
                settings.translation = TranslationConfig(**translation_config)
            
            if 'file' in config_data:
                file_config = config_data['file']
                settings.file_config = FileConfig(**file_config)
            
            if 'logging' in config_data:
                logging_config = config_data['logging']
                settings.logging = LoggingConfig(**logging_config)
        
        except Exception as e:
            print(f"Warning: Failed to load config file {config_path}: {e}")
    
    return settings


def save_settings(settings: Settings, config_path: Path) -> None:
    """
    Save settings to configuration file.
    
    Args:
        settings: Settings object to save
        config_path: Path to save configuration
    """
    config_data = {
        'model': {
            'provider': settings.model_provider,
            'name': settings.model_name
        },
        'api': {
            'ollama_url': settings.ollama_url,
            'openai_api_key': settings.openai_api_key,
            'openai_base_url': settings.openai_base_url,
            'claude_api_key': settings.claude_api_key,
            'qwen_api_key': settings.qwen_api_key,
            'qwen_base_url': settings.qwen_base_url
        },
        'translation': settings.translation.dict(),
        'file': settings.file_config.dict(),
        'logging': settings.logging.dict()
    }
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    except Exception as e:
        raise RuntimeError(f"Failed to save config file {config_path}: {e}")


def get_default_config_path() -> Path:
    """Get default configuration file path."""
    return Path.cwd() / "config" / "translation.yaml"
