"""Configuration package for PyARPES.

This package manages global runtime settings, workspace paths, and
matplotlib/jupyter integration for PyARPES. It provides:

- ConfigManager: central configuration controller
- WorkspaceManager: context-based workspace switcher
- get_config_manager: accessor for the global singleton ConfigManager

You typically don't need to instantiate ConfigManager yourself.
Use `get_config_manager()` to obtain the shared instance.
"""

from .interface import get_config_manager
from .manager import ConfigManager, should_initialize_automatically
from .workspace import WorkspaceManager

__all__ = [
    "ConfigManager",
    "WorkspaceManager",
    "get_config_manager",
    "should_initialize_automatically",
]
