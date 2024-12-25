"""Math snippets used elsewhere in PyARPES."""

from __future__ import annotations

from typing import Literal, TypedDict, TypeVar, Unpack

import numpy as np
import scipy.ndimage
import xarray as xr
from numpy.typing import NDArray

from arpes.constants import K_BOLTZMANN_EV_KELVIN


class ShiftParam(TypedDict, total=False):
    """Keyword parameter for scipy.ndimage.shift."""

    order: int
    mode: Literal[
        "reflect",
        "grid-mirror",
        "constant",
        "grid-constant",
        "nearest",
        "mirror",
        "grid-wrap",
        "wrap",
    ]
    cval: float
    prefilter: bool


def polarization(up: NDArray[np.float64], down: NDArray[np.float64]) -> NDArray[np.float64]:
    """The equivalent normalized difference for a two component signal."""
    return (up - down) / (up + down)


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
    assert axis != by_axis
    arr_copy = arr.copy()
    if isinstance(value, xr.DataArray):
        value = value.values
    assert isinstance(value, np.ndarray)
    for axis_idx in range(arr.shape[by_axis]):
        slc = (slice(None),) * by_axis + (axis_idx,) + (slice(None),) * (arr.ndim - by_axis - 1)
        shift_amount = (0,) * axis + (value[axis_idx],) + (0,) * (arr.ndim - axis - 1)
        shift_amount = shift_amount[1:] if axis > by_axis else shift_amount[:-1]
        arr_copy[slc] = scipy.ndimage.shift(arr[slc], shift_amount, **kwargs)
    return arr_copy


T = TypeVar("T", NDArray[np.float64], float)


def inv_fermi_distribution(
    energy: T,
    temperature: float,
    mu: float = 0.0,
) -> T:
    """Expects energy in eV and temperature in Kelvin."""
    return np.exp((energy - mu) / (K_BOLTZMANN_EV_KELVIN * temperature)) + 1.0


def fermi_distribution(
    energy: T,
    temperature: float,
) -> T:
    """Expects energy in eV and temperature in Kelvin."""
    return 1.0 / inv_fermi_distribution(energy, temperature)
