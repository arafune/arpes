"""Top-level initialization for the PyARPES package.

This module defines the public version of the package and triggers initial configuration
via `config.initialize()`. It serves as the entry point for importing and setting up
the PyARPES environment.

Key behaviors:
- Sets the package version via the `VERSION` string from `arpes.setting`.
- Exposes `__version__` in the public API (`__all__`).
- Initializes user configuration and plugin loading via `config.initialize()`.

Typical usage:
    import arpes
    print(arpes.__version__)

Note:
This module is automatically executed when `import arpes` is called.
"""

# pyright: reportUnusedImport=false
from __future__ import annotations

from pathlib import Path

from arpes import config

from .setting import VERSION

# Use both version conventions for people's sanity.

__version__ = VERSION
__all__ = ["__version__"]


config.initialize()
