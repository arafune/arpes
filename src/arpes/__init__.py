"""Top level module for PyARPES."""

# pyright: reportUnusedImport=false
from __future__ import annotations

from pathlib import Path

from arpes import config

from .setting import VERSION

# Use both version conventions for people's sanity.

__version__ = VERSION
__all__ = ["__version__"]


config.initialize()
