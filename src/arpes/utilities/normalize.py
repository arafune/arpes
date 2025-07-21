"""Utilities to programmatically get access to an ARPES spectrum as an xr.DataArray."""

from __future__ import annotations

import inspect
import warnings
from logging import DEBUG, INFO
from typing import TYPE_CHECKING

import xarray as xr

from arpes.debug import setup_logger

if TYPE_CHECKING:
    from xarray.core.common import DataWithCoords

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


__all__ = ("normalize_to_spectrum",)


def normalize_to_spectrum(data: DataWithCoords) -> xr.DataArray:
    """Tries to extract the actual ARPES spectrum from a dataset containing other variables."""
    if isinstance(data, xr.DataArray) and data.name == "spectrum":
        return data
    if isinstance(data, xr.Dataset):
        logger.debug(f"inspect.stack(): {inspect.stack()}")
        msg = "You use Dataset as a argument of "
        msg += f"{inspect.stack()[1].function} in {inspect.stack()[1].filename}\n"
        msg += "Remember to use a DataArray not a Dataset, "
        msg += "attempting to extract spectrum and copy attributes.\n"
        warnings.warn(
            msg,
            stacklevel=2,
        )
        if "up" in data.data_vars:
            assert isinstance(data.up, xr.DataArray)
            return data.up
        return data.S.spectrum
    assert isinstance(data, xr.DataArray)
    return data
