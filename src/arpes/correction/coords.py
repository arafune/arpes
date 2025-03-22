"""This module provides functions to manipulate coordinates in xarray DataArrays."""

from __future__ import annotations

import warnings
from logging import DEBUG, INFO
from typing import TYPE_CHECKING, LiteralString, get_args

import numpy as np
import xarray as xr

from arpes._typing import (
    CoordsOffset,
)
from arpes.debug import setup_logger

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from numpy.typing import NDArray

    from arpes.provenance import Provenance

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


__all__ = (
    "adjust_coords_to_limit",
    "corrected_coords",
    "extend_coords",
    "is_equally_spaced",
    "shift_by",
)


def adjust_coords_to_limit(
    da: xr.DataArray,
    new_limits: Mapping[Hashable, float],
) -> dict[Hashable, NDArray[np.float64]]:
    """Extend the coordinates of an xarray DataArray to given values for each dimension.

    The extension will ensure that the new coordinates cover up to the given extension value,
    and only the newly added coordinates will be returned.

    Parameters:
    da : xr.DataArray
        The original DataArray with equidistant coordinates.
    extensions : dict
        A dictionary specifying the values to which each coordinate should be extended.
        Example: {"x": 5, "y": -1}

    Returns:
        dict: A dictionary with the new extended coordinates for each dimension.
            Only the newly added coordinates are returned, which will be used in stretch_coords.
    """
    new_coords_dict = {}

    for dim, new_limit in new_limits.items():
        coords = da.coords[dim].values

        diffs = np.diff(coords)
        step = np.median(diffs)

        min_coord = np.min(coords)
        max_coord = np.max(coords)

        if new_limit > max_coord:
            new_coords = np.arange(max_coord + step, new_limit + step, step)
        elif new_limit < min_coord:
            new_coords = np.arange(new_limit, min_coord, step)
        else:
            new_coords = np.array([])

        new_coords_dict[dim] = new_coords

    return new_coords_dict


def extend_coords(
    da: xr.DataArray,
    new_coords: Mapping[Hashable, list[float] | NDArray[np.float64]],
) -> xr.DataArray:
    """Expand the coordinates of an xarray DataArray by adding new coordinate values.

    The new values will be filled with NaN.

    Parameters:
    da : xr.DataArray
        The original DataArray.
    new_coords : dict
        Dictionary where keys are coordinate names and values are lists of new coordinate values.
        If no new coordinates are specified, existing coordinates are retained.

    Returns:
        xr.DataArray: A new DataArray with expanded coordinates and NaN-filled missing values.
    """
    stretch_coords = {dim: da.coords[dim].values for dim in da.dims}

    for dim, values in new_coords.items():
        stretch_coords[dim] = np.union1d(stretch_coords.get(dim, []), values)

    shape = [len(stretch_coords[dim]) for dim in da.dims]
    coords = da.coords.copy()
    coords.update(stretch_coords)
    padding_value = 0 if da.dtype == np.int_ else np.nan

    expanded_da = xr.DataArray(
        np.full(shape, padding_value, dtype=np.float64),
        coords=coords,
        dims=list(da.dims),
        attrs=da.attrs,
    )
    expanded_da.loc[{dim: da.coords[dim] for dim in da.dims}] = da.astype(np.float64)

    return expanded_da


def is_equally_spaced(coords: NDArray[np.float64], tolerance: float = 1e-5) -> np.bool:
    """Check if the given coordinates are equally spaced within a given tolerance.

    Parameters:
    coords : np.ndarray
        The coordinates array to check.
    tolerance : float
        The acceptable tolerance for the spacing difference.

    Returns:
        bool: True if the coordinates are equally spaced within the tolerance, False otherwise.
    """
    diffs: NDArray[np.float64] = np.diff(coords)

    first_diff = diffs[0]

    return np.all(np.abs(diffs - first_diff) <= tolerance)


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

    corrected_data = data.copy(deep=True)

    for correction_type in correction_types:
        assert correction_type in get_args(CoordsOffset)

        if "_offset" in correction_type:
            coord_name: LiteralString = correction_type.split("_offset")[0]

            if coord_name not in corrected_data.coords:
                warnings.warn(
                    f"{coord_name} has not been set, while you correct "
                    f"{coord_name} by {correction_type}.",
                    stacklevel=2,
                )
            shift_value = (
                -corrected_data.attrs[correction_type]
                if coord_name in data.dims
                else corrected_data.attrs[correction_type]
            )
            corrected_data = shift_by(corrected_data, coord_name, shift_value)

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
