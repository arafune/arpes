"""Specialized type annotations for use in PyARPES.

In particular, `DataType` refers to either an xarray.DataArray or xarray.Dataset

`NormalizableDataType` referes to anything that can be tuned into datase,
such as by loading from the cache using an ID.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    TypedDict,
)

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.colors import Colormap


class _InteractiveConfigSettings(TypedDict, total=True):
    main_width: float
    marginal_width: float
    palette: str | Colormap


class WorkSpaceType(TypedDict):
    """TypedDict for arpes.CONFIG["WORKSPACE"]."""

    path: str | Path
    name: str


class ConfigSettings(TypedDict, total=True):
    """TypedDict for arpes.SETTINGS."""

    interactive: _InteractiveConfigSettings
    use_tex: bool
