"""This module provides functions to manipulate coordinates in xarray DataArrays."""

from __future__ import annotations

import warnings
from logging import DEBUG, INFO
from typing import TYPE_CHECKING, get_args

import xarray as xr

from arpes._typing import (
    CoordsOffset,
)
from arpes.debug import setup_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

    from arpes.provenance import Provenance

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


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
    assert coord_name in data.coords
    shifted_coords = {coord_name: data.coords[coord_name] + shift_value}
    shifted_data = data.assign_coords(**shifted_coords)
    provenance_: Provenance = shifted_data.attrs.get("provenance", {})
    provenance_shift_coords = provenance_.get("shift_coords", [])
    provenance_shift_coords.append((coord_name, shift_value))
    provenance_["shift_coords"] = provenance_shift_coords
    shifted_data.attrs["provenance"] = provenance_
    return shifted_data


def corrected_coords(
    data: xr.DataArray,
    correction_types: CoordsOffset | Sequence[CoordsOffset],
) -> xr.DataArray:
    """Corrects the coordinates of the given data by applying necessary transformations.

    Args:
        data (xr.DataArray): The input ARPES data array with coordinates to be corrected.
        correction_types (CoordsOffset | tuple[CoordsOffset]): Correction types to be applied to the
            data.

    Returns:
        xr.DataArray: The data array with corrected coordinates.
    """
    if isinstance(correction_types, str):
        correction_types = (correction_types,)
    assert isinstance(correction_types, tuple)

    corrected_data = data.copy(deep=True)

    for correction_type in correction_types:
        assert correction_type in get_args(CoordsOffset)

        if "_offset" in correction_type:
            coord_name = correction_type.split("_offset")[0]

            if coord_name not in corrected_data.coords:
                warnings.warn(
                    f"{coord_name} has not been set, while you correct "
                    f"{coord_name} by {correction_type}.",
                    stacklevel=2,
                )
                continue
            shift_value = (
                -corrected_data.attrs[correction_type]
                if coord_name in data.dims
                else corrected_data.attrs[correction_type]
            )
            corrected_data = shift_by(corrected_data, coord_name, shift_value)

            # data.attrs[coords_name] should consistent with data.coords[coords_name]
            if coord_name in corrected_data.attrs:
                corrected_data.attrs[coord_name] -= corrected_data.attrs[correction_type]

        # angle correction by beta or theta
        elif correction_type in {"beta", "theta"}:
            corrected_data = _apply_beta_theta_offset(corrected_data, correction_type)
        corrected_data.attrs[correction_type] = 0

        # provenance
        provenance_: Provenance = corrected_data.attrs.get("provenance", {})
        provenance_corrected_cords: list[CoordsOffset] = provenance_.get("coords_correction", [])
        provenance_corrected_cords.append(correction_type)
        provenance_["coords_correction"] = provenance_corrected_cords
        corrected_data.attrs["provenance"] = provenance_

    return corrected_data


def _apply_beta_theta_offset(
    data: xr.DataArray,
    correction_type: str,
) -> xr.DataArray:
    assert correction_type in {"beta", "theta"}
    axis = "psi" if data.S.is_slit_vertical else "phi"
    if correction_type == "beta":
        axis = "phi" if data.S.is_slit_vertical else "psi"
    data = shift_by(data, axis, data.attrs.get(correction_type, 0))
    data.attrs[correction_type] = 0
    data.coords[correction_type] = 0
    return data
