"""Implements a simple loader for Igor files.

This does not load data according to the PyARPES data model, so you should
ideally use a specific data loader where it is available.
"""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING, ClassVar

import xarray as xr

from arpes.debug import setup_logger
from arpes.endstations import (
    SingleFileEndstation,
)
from arpes.load_pxt import read_single_pxt

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from _typeshed import Incomplete

    from arpes._typing import ScanDesc, Spectrometer


__all__ = ("IgorEndstation",)


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


class IgorEndstation(SingleFileEndstation):
    """A generic file loader for PXT files.

    This makes no assumptions about whether data is from a hemisphere
    or otherwise, so it might not be perfect for all Igor users, but it
    is a place to start and to demonstrate how to implement a data loading
    plugin.
    """

    PRINCIPAL_NAME = "Igor"
    ALIASES: ClassVar[list[str]] = [
        "IGOR",
        "pxt",
        "pxp",
        "Wave",
        "wave",
    ]

    _TOLERATED_EXTENSIONS: ClassVar[set[str]] = {
        ".pxt",
    }
    _SEARCH_PATTERNS: tuple[str, ...] = (
        r"[\-a-zA-Z0-9_\w]+_[0]+{}$",
        r"[\-a-zA-Z0-9_\w]+_{}$",
        r"[\-a-zA-Z0-9_\w]+{}$",
        r"[\-a-zA-Z0-9_\w]+[0]{}$",
    )

    RENAME_KEYS: ClassVar[dict[str, str]] = {}

    MERGE_ATTRS: ClassVar[Spectrometer] = {}

    ATTR_TRANSFORMS: ClassVar[dict[str, Callable[..., dict[str, float | list[str] | str]]]] = {}

    def load_single_frame(
        self,
        frame_path: str | Path = "",
        scan_desc: ScanDesc | None = None,
        **kwargs: Incomplete,
    ) -> xr.Dataset:
        """Igor .pxt and .ibws are single files so we just read the one passed here."""
        del kwargs
        logger.info(f"frame_path: {frame_path}, scan_desc: {scan_desc}")

        pxt_data = read_single_pxt(frame_path)
        return xr.Dataset({"spectrum": pxt_data}, attrs=pxt_data.attrs)
