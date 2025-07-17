"""Configuration access for the global ConfigManager singleton in PyARPES.

This module provides:

- A single function `get_config_manager()` to access the global ConfigManager instance safely and
    lazily.
- Public accessor functions for common runtime configuration values, workspace paths, and
  environment flags.

Why this module exists:
-----------------------
- Avoids circular import issues by isolating ConfigManager instantiation.
- Ensures a consistent, shared configuration environment across modules, scripts, and notebooks.
- Provides a stable and simplified API for accessing runtime settings.

Usage:
------
>>> from arpes.configuration.access import get_config_manager, get_workspace_path
>>> config = get_config_manager()
>>> workspace = get_workspace_path()

This pattern maintains modularity, testability, and ease of interactive use.
"""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Any

from arpes.configuration.manager import ConfigManager, should_initialize_automatically
from arpes.debug import setup_logger

if TYPE_CHECKING:
    from pathlib import Path

# Internal singleton instance holder
_config_manager_instance: ConfigManager | None = None


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


def get_config_manager() -> ConfigManager:
    """Return the singleton instance of ConfigManager.

    This function lazily creates the ConfigManager instance on first use.
    If it hasn't been initialized yet, it will automatically initialize it
    when the environment is Jupyter or marimo.

    Returns:
        ConfigManager: The global configuration manager instance.

    Notes:
        - For non-interactive environments (e.g., scripts), users may call
          `get_config_manager().initialize()` explicitly if needed.
        - For Jupyter and marimo, initialization is automatic.
    """
    global _config_manager_instance  # noqa: PLW0603
    logger.debug("Accessing global ConfigManager instance")
    logger.debug(f"_config_manager_instance: {_config_manager_instance}")
    if _config_manager_instance is None:
        _config_manager_instance = ConfigManager()
        if should_initialize_automatically():
            logger.debug("is jupyter or is_marimo")
            _config_manager_instance.initialize()
    return _config_manager_instance


def get_workspace_path() -> Path:
    """Return the absolute path to the current workspace directory.

    Returns:
        Path: The full path to the workspace root directory.
    """
    return get_config_manager().workspace_path


def get_workspace_name() -> str:
    """Return the name of the current workspace.

    Typically, this is the folder name of the workspace directory.

    Returns:
        str: The workspace name.
    """
    return str(get_config_manager().workspace_name)


def get_data_path() -> str | Path | None:
    """Return the path to the data directory under the workspace.

    This directory typically contains measurement data files.

    Returns:
        Path | None: The path to the data folder, or None if not set.
    """
    return get_config_manager().data_path


def get_dataset_path() -> Path | None:
    """Return the path to the dataset directory under the workspace.

    This folder typically contains processed or structured datasets.

    Returns:
        Path | None: The path to the datasets folder, or None if not set.
    """
    return get_config_manager().dataset_path


def get_figure_path() -> Path | None:
    """Return the path to the figure output directory under the workspace.

    This directory is used for saving generated plots and figures.

    Returns:
        Path | None: The path to the figure output folder, or None if not set.
    """
    return get_config_manager().figure_path


def get_logging_file() -> str | Path | None:
    """Return the path to the active IPython or session log file.

    This is populated when logging is enabled during initialization.

    Returns:
        str | Path | None: The log file path, or None if logging not started.
    """
    return get_config_manager().config.get("LOGGING_FILE")


def get_logging_started() -> bool:
    """Check whether logging has been started for the session.

    Returns:
        bool: True if logging was initialized, otherwise False.
    """
    return bool(get_config_manager().config.get("LOGGING_STARTED", False))


def get_config_value(key: str, default: Any = None) -> Any:  # noqa: ANN401
    """Get a value from the top-level config dictionary.

    Args:
        key (str): The config key to retrieve.
        default (Any): Default value to return if the key is missing.

    Returns:
        Any: The value of the requested config item.
    """
    return get_config_manager().config.get(key, default)


def get_settings_value(key: str, default: Any = None) -> Any:  # noqa: ANN401
    """Get a value from the top-level settings dictionary.

    Args:
        key (str): The setting key to retrieve.
        default (Any): Default value to return if the key is missing.

    Returns:
        Any: The value of the requested setting item.
    """
    return get_config_manager().settings.get(key, default)


def get_full_config() -> dict:
    """Return the entire runtime configuration dictionary.

    Returns:
        dict: The full config dictionary (read-only usage recommended).
    """
    return get_config_manager().config


def get_full_settings() -> dict:
    """Return the entire user settings dictionary.

    Returns:
        dict: The full settings dictionary (read-only usage recommended).
    """
    return get_config_manager().settings
