"""Global variables for PyARPES.

- Utility functions for updating internal paths

All stateful settings are collected here to avoid circular imports
and to allow safe initialization from a central place.
"""

from __future__ import annotations

from pathlib import Path

# Base paths
SOURCE_ROOT: str = str(Path(__file__).resolve().parent)

VERSION = "5.0.0"
