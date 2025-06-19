"""This module provides functions to manipulate intensity map (values) in xarray DataArrays."""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import numpy as np
import scipy.ndimage
import xarray as xr

from arpes.constants import TWO_DIMENSION
from arpes.debug import setup_logger

from . import coords

if TYPE_CHECKING:
    from numpy.typing import NDArray


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


__all__ = ("shift", "shift_by")


class ShiftParam(TypedDict, total=False):
    """Keyword parameter for scipy.ndimage.shift."""

    order: Literal[0, 1, 2, 3, 4, 5]
    mode: Literal[
        "reflect",
        "grid-mirror",
        "constant",  # default , use cval
        "grid-constant",
        "nearest",
        "mirror",
        "grid-wrap",
        "wrap",
    ]
    cval: float  # default is 0.0
    prefilter: bool


def shift(  # noqa: PLR0913
    data: xr.DataArray,
    other: xr.DataArray | NDArray[np.float64],
    shift_axis: str = "",
    by_axis: str = "",
    *,
    extend_coords: bool = False,
    shift_coords: bool = False,
) -> xr.DataArray:
    """Shifts the data along the specified axis, used by G.shift_by.

    Currently, only supports shifting by a one-dimensional array.

    Args:
        data (xr.DataArray): The xr.DataArray to shift.
        other (xr.DataArray | NDArray): Data to shift by. Only supports one-dimensional array.
        shift_axis (str): The axis to shift along, which is 1D.
        by_axis (str): The dimension name of `other`. Ignored when `other` is an xr.DataArray.
        extend_coords (bool): If True, the coords expands.  Default is False.
        shift_coords (bool): Whether to shift the coordinates as well.
            The arg will be removed, because it is not unique way to shift from the "other".
            Currently it uses mean value of "other".

    Returns:
        xr.DataArray: The shifted xr.DataArray.

    Todo:
        - Add tests.Data shift along the axis.

    Note:
        zero_nans is removed.  Use DataArray.fillna(0), if needed.
    """
    assert shift_axis, "shift_by must take shift_axis argument."
    shift_amount, mean_shift, by_axis = _compute_shift_amount(
        data=data,
        other=other,
        shift_axis=shift_axis,
        by_axis=by_axis,
        shift_coords=shift_coords,
    )
    shift_amount_physical_axis = -other

    if extend_coords:
        if np.min(shift_amount_physical_axis) < 0:
            extended_coord = coords.adjust_coords_to_limit(
                da=data,
                new_limits={
                    shift_axis: data.coords[shift_axis].min().item()
                    + np.min(shift_amount_physical_axis),
                },
            )
            data = coords.extend_coords(data, new_coords=extended_coord)
        if np.max(shift_amount_physical_axis) > 0:
            extended_coord = coords.adjust_coords_to_limit(
                da=data,
                new_limits={
                    shift_axis: data.coords[shift_axis].max().item()
                    + np.max(shift_amount_physical_axis),
                },
            )
            data = coords.extend_coords(da=data, new_coords=extended_coord)

    padding_value = 0 if data.dtype == np.int_ else np.nan
    shifted_data: NDArray[np.float64] = shift_by(
        arr=data.values,
        value=shift_amount,
        axis=data.dims.index(shift_axis),
        by_axis=data.dims.index(by_axis),
        order=1,
        mode="constant",
        cval=padding_value,
    )
    built_data = data.G.with_values(shifted_data)
    if shift_coords:
        built_data = built_data.assign_coords(
            {shift_axis: data.coords[shift_axis] + mean_shift},
        )
    return built_data


def _compute_shift_amount(
    data: xr.DataArray,
    other: xr.DataArray | NDArray[np.float64],
    shift_axis: str,
    by_axis: str = "",
    *,
    shift_coords: bool = False,
) -> tuple[NDArray[np.float64], float, str]:
    """Compute shift amount based on `other` and determine `by_axis` if necessary.

    Helper function for `shift`

    Args:
        data (xr.DataArray): The target DataArray to shift.
        other (xr.DataArray | NDArray): Shift values (must be 1D).
        shift_axis (str): The axis to shift along.
        by_axis (str, optional): The dimension name of `other`.
            If empty, inferred for `np.ndarray`.
        shift_coords (bool, optional): Whether to adjust the coordinates based on the mean
            shift.

    Returns:
        tuple[NDArray[np.float64], float, str]:
            - shift_amount: The computed shift values.
            - mean_shift: The mean value of `other` (0 if not shifting coords).
            - by_axis: The determined `by_axis` name.
    """
    assert other.ndim == 1, "`other` must be a 1D array."

    mean_shift: float = 0.0

    if isinstance(other, xr.DataArray):
        by_axis = str(other.dims[0])
        assert len(other.coords[by_axis]) == len(data.coords[by_axis]), (
            "Mismatch in coordinate length."
        )
        if shift_coords:
            mean_shift = float(np.mean(other.values))
            other = other - mean_shift
        shift_amount = -other.values / data.G.stride(generic_dim_names=False)[shift_axis]

    else:  # other is np.ndarray
        assert isinstance(other, np.ndarray)
        if not by_axis:
            if data.ndim == TWO_DIMENSION:
                by_axis = str(set(data.dims).difference({shift_axis}).pop())
                logger.debug(f"Using {by_axis} as by_axis for shift.")
            else:
                msg = 'When np.ndarray is used as `other`, "by_axis" is required.'
                raise TypeError(msg)
        logger.debug(f"Using {by_axis} as by_axis for shift.")
        assert other.shape[0] == len(data.coords[by_axis]), "Mismatch in coordinate length."
        if shift_coords:
            mean_shift = float(np.mean(other))
            other = other - mean_shift
        shift_amount = -other / data.G.stride(generic_dim_names=False)[shift_axis]
    return shift_amount, mean_shift, by_axis


def shift_by(
    arr: NDArray[np.float64],
    value: NDArray[np.float64],
    axis: int = 0,
    by_axis: int = 0,
    **kwargs: Unpack[ShiftParam],
) -> NDArray[np.float64]:
    """Shifts slices of `arr` perpendicular to `by_axis` by `value`.

    Args:
        arr (NDArray[np.float64): Input array to be shifted.
        value (NDArray[np.float64): Array of shift values.
        axis (int): Axis number of np.ndarray for shift.
        by_axis (int): Axis number of np.ndarray for non-shift.
        **kwargs(ShiftParam): Additional parameters to pass to scipy.ndimage.shift.

    Returns:
        NDArray[np.float64]: The shifted array.
    """
    assert axis != by_axis, "`axis` and `by_axis` must be different."
    arr_copy = arr.copy()
    assert isinstance(value, np.ndarray)
    assert value.shape == (arr.shape[by_axis],), (
        "`value` must have the same length as `arr` along `by_axis`."
    )
    for axis_idx in range(arr.shape[by_axis]):
        slc = (slice(None),) * by_axis + (axis_idx,) + (slice(None),) * (arr.ndim - by_axis - 1)
        shift_amount = (0,) * axis + (value[axis_idx],) + (0,) * (arr.ndim - axis - 1)
        shift_amount = shift_amount[1:] if axis > by_axis else shift_amount[:-1]
        logger.debug(f"Shifting slice {slc} by {shift_amount}")
        arr_copy[slc] = scipy.ndimage.shift(arr[slc], shift_amount, **kwargs)
    return arr_copy
