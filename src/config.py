"""Configuration management for SubtitleForge Pro."""
import json
import os
from pathlib import Path
from typing import Any, Optional


def setup_huggingface_cache():
    """Setup Hugging Face cache directory to avoid permission issues on Windows."""
    hf_cache = Path.home() / '.cache' / 'hf_home'
    hf_cache.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(hf_cache)
    os.environ['HF_DATASETS_CACHE'] = str(hf_cache / 'datasets')
    os.environ['TRANSFORMERS_CACHE'] = str(hf_cache / 'transformers')


setup_huggingface_cache()


class Config:
    """Simple configuration manager with local file storage."""

    def __init__(self):
        self.config_dir = Path.home() / '.subtitleforge'
        self.config_file = self.config_dir / 'config.json'
        self._config: dict[str, Any] = {}
        self._load()

    def _load(self):
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, encoding='utf-8') as f:
                    self._config = json.load(f)
            except Exception:
                self._config = {}
        else:
            self._config = {}

    def _save(self):
        """Save configuration to file."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self._config, f, indent=2, ensure_ascii=False)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self._config.get(key, default)

    def set(self, key: str, value: Any):
        """Set a configuration value and save."""
        self._config[key] = value
        self._save()

    def get_api_key(self) -> Optional[str]:
        """Get the cached Gemini API key."""
        return self._config.get('gemini_api_key')

    def set_api_key(self, api_key: str):
        """Save the Gemini API key."""
        self._config['gemini_api_key'] = api_key
        self._save()

    def clear_api_key(self):
        """Clear the cached API key."""
        if 'gemini_api_key' in self._config:
            del self._config['gemini_api_key']
            self._save()


# Global config instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get the global config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
