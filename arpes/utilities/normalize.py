"""Utilities to programmatically get access to an ARPES spectrum as an xr.DataArray."""

from __future__ import annotations

from typing import TYPE_CHECKING

import xarray as xr

if TYPE_CHECKING:
    from arpes._typing import DataType

__all__ = (
    "normalize_to_spectrum",
    "normalize_to_dataset",
)


def normalize_to_spectrum(data: xr.DataArray | xr.Dataset | str) -> xr.DataArray:
    """Tries to extract the actual ARPES spectrum from a dataset containing other variables."""
    import arpes.xarray_extensions  # noqa: F401
    from arpes.io import load_data

    if isinstance(data, xr.Dataset):
        if "up" in data.data_vars:
            return data.up
        return data.S.spectrum
    if isinstance(data, str):
        return normalize_to_spectrum(load_data(data))
    return data


def normalize_to_dataset(data: DataType | str | int) -> xr.Dataset | None:
    """Loads data if we were given a path instead of a loaded data sample."""
    from arpes.io import load_data

    if isinstance(data, xr.Dataset):
        return data
    if isinstance(data, str | int):
        return load_data(data)
    return None
