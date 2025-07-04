"""Runtime level configuration for PyARPES.

For the most part, this contains state related to:

1. UI settings
2. TeX and plotting
3. Data loading
4. Workspaces

This module also provides functions for loading configuration
in via external files, to allow better modularity between
different projects.
"""

from __future__ import annotations

from arpes.configuration.manager import config_manager as _manager

__all__ = [
    "config",
    "initialize",
    "is_using_tex",
    "load_plugins",
    "settings",
    "setup_logging",
    "use_tex",
    "workspace_name",
    "workspace_path",
]

# Aliases for backward compatibility (optional, deprecate later)
config = _manager.config
settings = _manager.settings

initialize = _manager.initialize
setup_logging = _manager.setup_logging
use_tex = _manager.use_tex
is_using_tex = _manager.is_using_tex
load_plugins = _manager.load_plugins
workspace_path = _manager.workspace_path
workspace_name = _manager.workspace_name
