"""Global configuration state for PyARPES.

This module defines global runtime state used throughout the library,
including:

- File system paths (source root, data folder)
- Mutable configuration dictionaries (CONFIG, SETTINGS)
- Global unit registry
- Utility functions for updating internal paths

All stateful settings are collected here to avoid circular imports
and to allow safe initialization from a central place.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arpes._typing import ConfigSettings, ConfigType

# Base paths
SOURCE_ROOT: str = str(Path(__file__).resolve().parent)
DATA_PATH: str | None = None  # Can be set explicitly by user or loader
FIGURE_PATH: str | Path | None = None
DATASET_PATH: str | Path | None = None

VERSION = "5.0.0-pre4"

# Global settings for GUI, plotting, etc.
SETTINGS: ConfigSettings = {
    "interactive": {
        "main_width": 350,
        "marginal_width": 150,
        "palette": "magma",
    },
    "use_tex": False,
}

# Global runtime configuration (project/workspace specific)
CONFIG: ConfigType = {
    "WORKSPACE": {"path": "", "name": ""},
    "CURRENT_CONTEXT": None,
    "ENABLE_LOGGING": True,
    "LOGGING_STARTED": False,
    "LOGGING_FILE": None,
}


def update_paths(user_path: str | Path) -> None:
    """Set standard paths like DATA_PATH and figure directories.

    This function is invoked when configuration is initialized via
    `update_configuration` to populate figure/dataset locations.

    Args:
        user_path (str | Path): The base user path, usually a workspace root.
    """
    global DATA_PATH  # noqa: PLW0603

    path = Path(user_path).resolve() if user_path else None
    if path:
        CONFIG["WORKSPACE"]["path"] = str(path)
        CONFIG["WORKSPACE"]["name"] = path.name
        DATA_PATH = str(path / "data")
