"""Top-level initialization for the PyARPES package.

This module defines the public version of the package and triggers initial configuration
via `arpes.initialize()`. It serves as the entry point for importing and setting up
the PyARPES environment.

Key behaviors:
- Sets the package version via the `VERSION` string from `arpes.setting`.
- Exposes `__version__` in the public API (`__all__`).
- Automatically initializes configuration in interactive environments (Jupyter, marimo).
- Provides a manual `initialize()` function for script-based use.

Typical usage:
    import arpes
    arpes.initialize()  # Optional; only needed in standalone scripts.

Note:
This module is automatically executed when `import arpes` is called.
"""

from __future__ import annotations

from arpes.configuration import get_config_manager, should_initialize_automatically
from arpes.setting import VERSION

__version__ = VERSION
__all__ = ["__version__"]


def initialize() -> None:
    """Manually initialize PyARPES configuration, logging, and plugin system.

    This is useful when running in scripts or command-line environments
    where automatic initialization (Jupyter/marimo) is not triggered.
    """
    get_config_manager()


# Automatically initialize if in Jupyter or marimo environment
if should_initialize_automatically():
    initialize()
