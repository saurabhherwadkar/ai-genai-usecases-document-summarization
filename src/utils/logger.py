# Logger module - configures structured logging from YAML configuration file.
# Provides a factory function to get named loggers throughout the application.

import logging
import logging.config
from pathlib import Path

import yaml


def _setup_logging() -> None:
    """Load and apply the YAML logging configuration.

    Reads the logging.yaml file from the config directory and applies it
    to the Python logging system. Creates the logs directory if it does not exist.
    """
    # Determine the logging config file path relative to the project root
    config_path = Path(__file__).parent.parent.parent / "config" / "logging.yaml"

    # Ensure the logs directory exists for file handlers
    logs_dir = Path(__file__).parent.parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Load and apply the YAML logging configuration
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as log_config_file:
            log_config = yaml.safe_load(log_config_file)
        logging.config.dictConfig(log_config)
    else:
        # Fall back to basic configuration if YAML file is missing
        logging.basicConfig(level=logging.INFO)


# Apply logging configuration on module import
_setup_logging()


def get_logger(name: str) -> logging.Logger:
    """Get a named logger instance with the application's logging configuration.

    Args:
        name: The logger name, typically the module's __name__.

    Returns:
        logging.Logger: A configured logger instance for the given name.
    """
    return logging.getLogger(name)
