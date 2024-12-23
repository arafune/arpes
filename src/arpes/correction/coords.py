"""This module provides functions to manipulate coordinates in xarray DataArrays."""

from __future__ import annotations

from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, get_args

import xarray as xr

from arpes._typing import (
    CoordsOffset,
)

if TYPE_CHECKING:
    from arpes.provenance import Provenance

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


def shift_by(
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
    shifted_data = data.assign_coords(**shifted_coords)
    provenance_: Provenance = shifted_data.attrs.get("provenance", {})
    provenance_shift_coords = provenance_.get("shift_coords", [])
    provenance_shift_coords.append((coord_name, shift_value))
    shifted_data.attrs["provenance"]["shift_coords"] = provenance_shift_coords
    return shifted_data


def corrected_coords(
    data: xr.DataArray, correction_types: CoordsOffset | tuple[CoordsOffset, ...]
) -> xr.DataArray:
    if isinstance(correction_types, str):
        correction_types = (correction_types,)
    assert isinstance(correction_types, tuple)
    for correction_type in correction_types:
        assert correction_type in get_args(CoordsOffset)
