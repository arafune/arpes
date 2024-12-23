"""This module provides functions to manipulate coordinates in xarray DataArrays."""

from __future__ import annotations

from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger

import xarray as xr

from arpes.provenance import Provenance, provenance, update_provenance

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


def shift_coord_by(
    data: xr.DataArray,
    coord_name: str,
    shift_value: float,
) -> xr.DataArray:
    """Shifts the coordinates by the specified values.

    Args:
        data (xr.DataArray): The DataArray to shift.
        coord_name (str): The coordinate name to shift.
        shift_value (float): The amount of the shift.

    Returns:
        xr.DataArray: The DataArray with shifted coordinates.
    """
    assert isinstance(data, xr.DataArray)
    assert coord_name in data.dims
    shifted_coords = {coord_name: data.coords[coord_name] + shift_value}
    return data.assign_coords(**shifted_coords)
