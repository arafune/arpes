"""Math snippets used elsewhere in PyARPES."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, TypedDict, Unpack

import numpy as np
import scipy.ndimage
import xarray as xr

from arpes.constants import K_BOLTZMANN_EV_KELVIN

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SHIFTPARAM(TypedDict, total=False):
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


def polarization(up: NDArray[np.float_], down: NDArray[np.float_]) -> NDArray[np.float_]:
    """The equivalent normalized difference for a two component signal."""
    return (up - down) / (up + down)


def shift_by(
    arr: NDArray[np.float_],
    value: xr.DataArray | NDArray[np.float_],
    axis: int = 0,
    by_axis: int = 0,
    **kwargs: Unpack[SHIFTPARAM],
) -> NDArray[np.float_]:
    """Shifts slices of `arr` perpendicular to `by_axis` by `value`.

    Args:
        arr ([TODO:type]): [TODO:description]
        value ([TODO:type]): [TODO:description]
        axis (int): Axis number of np.ndarray for shift
        by_axis (int): Axis number of np.ndarray for non-shift
        **kwargs(SHIFTPARAM): pass to scipy.ndimage.shift
    """
    assert axis != by_axis
    arr_copy = arr.copy()
    if isinstance(value, xr.DataArray):
        value = value.values
    assert isinstance(value, np.ndarray)
    if not isinstance(value, Iterable):
        value = list(itertools.repeat(value, times=arr.shape[by_axis]))
    for axis_idx in range(arr.shape[by_axis]):
        slc = (slice(None),) * by_axis + (axis_idx,) + (slice(None),) * (arr.ndim - by_axis - 1)
        shift_amount = (0,) * axis + (value[axis_idx],) + (0,) * (arr.ndim - axis - 1)
        shift_amount = shift_amount[1:] if axis > by_axis else shift_amount[:-1]
        arr_copy[slc] = scipy.ndimage.shift(arr[slc], shift_amount, **kwargs)
    return arr_copy


def inv_fermi_distribution(
    energy: NDArray[np.float_] | float,
    temperature: float,
    mu: float = 0.0,
) -> NDArray[np.float_]:
    """Expects energy in eV and temperature in Kelvin."""
    return np.exp((energy - mu) / (K_BOLTZMANN_EV_KELVIN * temperature)) + 1.0


def fermi_distribution(
    energy: NDArray[np.float_] | float,
    temperature: float,
) -> NDArray[np.float_] | float:
    """Expects energy in eV and temperature in Kelvin."""
    return 1.0 / inv_fermi_distribution(energy, temperature)
