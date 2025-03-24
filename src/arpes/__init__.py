"""Top level module for PyARPES."""

# pyright: reportUnusedImport=false
from __future__ import annotations

from pathlib import Path

# Use both version conventions for people's sanity.
VERSION = "5.0.0-pre2"
__version__ = VERSION


__all__ = ["__version__"]


SOURCE_ROOT = str(Path(__file__).parent)
DATA_PATH: str | None = None
HAS_LOADED: bool = False

if not HAS_LOADED:
    import arpes.config
