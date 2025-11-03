"""
Simple configuration management module.

Loads configuration from configs/settings.yml and sets up logging.
"""

import logging
import logging.config
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class ConfigManager:
    """Simple configuration manager."""

    def __init__(self):
        """Initialize configuration manager."""
        self._config: Optional[Dict[str, Any]] = None
        self._logger: Optional[logging.Logger] = None
        self.project_root = self._find_project_root()
        self.config_file = self.project_root / "configs" / "settings.yml"
        self.logging_config_file = self.project_root / "configs" / "logging_config.yml"

    def _find_project_root(self) -> Path:
        """Find project root directory."""
        current = Path.cwd()
        while current != current.parent:
            if (current / "configs").exists():
                return current
            current = current.parent
        return Path.cwd()

    @property
    def config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        if self._config is None:
            self._config = self._load_config()
        return self._config

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
            print(f"✓ Configuration loaded from {self.config_file}")
            return config_data
        except FileNotFoundError:
            print(f"⚠ Configuration file not found: {self.config_file}")
            return {}
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key (e.g., 'database.host')
            default: Default value if key not found
        """
        keys = key.split(".")
        value = self.config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    def setup_logging(self) -> logging.Logger:
        """Setup logging from configs/logging_config.yml."""
        if self._logger is not None:
            return self._logger

        try:
            if self.logging_config_file.exists():
                with open(self.logging_config_file, "r") as f:
                    logging_config = yaml.safe_load(f)
                logging.config.dictConfig(logging_config)
                self._logger = logging.getLogger(__name__)
                self._logger.info(f"Logging configured from {self.logging_config_file}")
            else:
                self._setup_basic_logging()
        except Exception as e:
            print(f"⚠ Error setting up logging: {e}")
            self._setup_basic_logging()

        return self._logger

    def _setup_basic_logging(self):
        """Setup basic logging configuration."""
        level = self.get("logging.level", "INFO")
        format_str = self.get(
            "logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format=format_str,
            handlers=[logging.StreamHandler()],
        )
        self._logger = logging.getLogger(__name__)
        self._logger.info("Basic logging configuration applied")

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Get a logger instance."""
        if self._logger is None:
            self.setup_logging()
        return logging.getLogger(name or __name__)


# Global instance
_config = ConfigManager()


def get_config() -> Dict[str, Any]:
    """Get configuration dictionary."""
    return _config.config


def get(key: str, default: Any = None) -> Any:
    """Get configuration value using dot notation."""
    return _config.get(key, default)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger instance."""
    return _config.get_logger(name)


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    return _config.setup_logging()


# Example usage
if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()

    # Access configuration
    app_name = get("app.name", "Unknown App")
    debug_mode = get("development.debug", False)

    logger.info(f"Application: {app_name}")
    logger.info(f"Debug mode: {debug_mode}")

    print("✅ Configuration module loaded successfully!")
