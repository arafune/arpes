"""Provides coordinate aware filters and smoothing."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from scipy import ndimage

from arpes.provenance import provenance

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = (
    "gaussian_filter_arr",
    "gaussian_filter",
    "boxcar_filter_arr",
    "boxcar_filter",
)


def gaussian_filter_arr(
    arr: xr.DataArray,
    sigma: dict[str, float | int] | None = None,
    repeat_n: int = 1,
    *,
    default_size: int = 1,
    use_pixel: bool = False,
) -> xr.DataArray:
    """Coordinate aware `scipy.ndimage.filters.gaussian_filter`.

    Args:
        arr(xr.DataArray): ARPES data
        sigma: Kernel sigma, specified in terms of axis units (if use_pixel is False).
          An axis that is not specified will have a kernel width of `default_size` in index units.
        repeat_n: Repeats n times.
        default_size: Changes the default kernel width for axes not specified in `sigma`.
          Changing this parameter and leaving `sigma` as None allows you to smooth with an
          even-width kernel in index-coordinates.
        use_pixel(bool): if True, the sigma value is specified by pixel units not axis units.

    Returns:
        Smoothed data.
    """
    if sigma is None:
        sigma = {}
    sigma = {k: int(v / (arr.coords[k][1] - arr.coords[k][0])) for k, v in sigma.items()}
    for dim in arr.dims:
        if dim not in sigma:
            sigma[dim] = default_size
    widths_pixel: tuple[int, ...] = tuple(sigma[k] for k in arr.dims)
    values = arr.values
    for _ in range(repeat_n):
        values = ndimage.gaussian_filter(values, widths_pixel)
    filtered_arr = xr.DataArray(values, arr.coords, arr.dims, attrs=copy.deepcopy(arr.attrs))
    if "id" in filtered_arr.attrs:
        del filtered_arr.attrs["id"]
        provenance(
            filtered_arr,
            arr,
            {
                "what": "Gaussian filtered data",
                "by": "gaussian_filter_arr",
                "sigma": sigma,
                "use_pixel": use_pixel,
            },
        )
    return filtered_arr


def boxcar_filter_arr(
    arr: xr.DataArray,
    size: dict[str, int | float] | None = None,
    repeat_n: int = 1,
    default_size: int = 1,
    *,
    use_pixel: bool = False,
) -> xr.DataArray:
    """Coordinate aware `scipy.ndimage.filters.boxcar_filter`.

    Args:
        arr: ARPES data
        size: Kernel size, specified in terms of axis units (if use_pixel is False).
              An axis that is not specified will have a kernel width of `default_size` in
              index units.
        repeat_n: Repeats n times.
        default_size: Changes the default kernel width for axes not
            specified in `sigma`. Changing this parameter and leaving
            `sigma` as None allows you to smooth with an even-width
            kernel in index-coordinates.
        use_pixel(bool): if True, the size value is specified by pixel (i.e. index) units,
            not axis (physical) units.

    Returns:
        smoothed data.
    """
    assert isinstance(arr, xr.DataArray)
    if size is None:
        size = {}
    size = {k: int(v / (arr.coords[k][1] - arr.coords[k][0])) for k, v in size.items()}
    for dim in arr.dims:
        if dim not in size:
            size[str(dim)] = default_size
    widths_pixel: tuple[int, ...] = tuple([size[str(k)] for k in arr.dims])
    array_values = np.nan_to_num(arr.values, copy=True)
    for _ in range(repeat_n):
        array_values = ndimage.uniform_filter(array_values, widths_pixel)
    filtered_arr = xr.DataArray(array_values, arr.coords, arr.dims, attrs=copy.deepcopy(arr.attrs))
    if "id" in arr.attrs:
        del filtered_arr.attrs["id"]
        provenance(
            filtered_arr,
            arr,
            {
                "what": "Boxcar filtered data",
                "by": "boxcar_filter_arr",
                "size": size,
                "use_pixel": use_pixel,
            },
        )
    return filtered_arr


def gaussian_filter(sigma: dict[str, float | int] | None = None, repeat_n: int = 1) -> Callable:
    """A partial application of `gaussian_filter_arr`.

    For further derivative analysis functions.

    Args:
        sigma(dict[str, float|int] | None): Kernel sigma
        repeat_n(int): Repeats n times.

    Returns:
        A function which applies the Gaussian filter.
    """

    def f(arr: xr.DataArray) -> xr.DataArray:
        return gaussian_filter_arr(arr, sigma, repeat_n)

    return f


def boxcar_filter(size: dict[str, int | float] | None = None, repeat_n: int = 1) -> Callable:
    """A partial application of `boxcar_filter_arr`.

    Output can be passed to derivative analysis functions.

    Args:
        size(dict[str | int, float]): Kernel size
        repeat_n(int):Repeats n times.

    Returns:
        A function which applies the boxcar.
    """

    def f(arr: xr.DataArray) -> xr.DataArray:
        return boxcar_filter_arr(arr, size, repeat_n)

    return f
