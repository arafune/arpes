"""Provides coordinate aware filters and smoothing."""

from collections.abc import Hashable
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.signal import savgol_filter

from arpes.provenance import Provenance, provenance, update_provenance
from arpes.utilities import normalize_to_spectrum

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = (
    "boxcar_filter_arr",
    "gaussian_filter_arr",
    "savgol_filter_multi",
    "savitzky_golay_filter",
)


def gaussian_filter_arr(
    arr: xr.DataArray,
    sigma: dict[Hashable, float | int] | None = None,
    iteration_n: int = 1,
    *,
    default_size: int = 1,
    use_pixel: bool = False,
) -> xr.DataArray:
    """Coordinate aware `scipy.ndimage.filters.gaussian_filter`.

    Args:
        arr(xr.DataArray): ARPES data
        sigma (dict[Hashable, int]): Kernel sigma, specified in terms of axis units.
          (if use_pixel is False).
          An axis that is not specified will have a kernel width of `default_size` in index units.
        iteration_n: Repeats n times.
        default_size: Changes the default kernel width for axes not specified in `sigma`.
          Changing this parameter and leaving `sigma` as None allows you to smooth with an
          even-width kernel in index-coordinates.
        use_pixel(bool): if True, the sigma value is specified by pixel units not axis units.

    Returns:
        Smoothed data.
    """
    sigma = sigma or {}
    sigma_pixel = (
        {k: int(v) for k, v in sigma.items()}
        if use_pixel
        else {k: int(v / (arr.coords[k][1] - arr.coords[k][0])) for k, v in sigma.items()}
    )
    for dim in arr.dims:
        if dim not in sigma_pixel:
            sigma_pixel[dim] = default_size
    widths_pixel: tuple[int, ...] = tuple(sigma_pixel[k] for k in arr.dims)
    values = arr.values
    for _ in range(iteration_n):
        values = ndimage.gaussian_filter(values, widths_pixel)
    filtered_arr = xr.DataArray(values, arr.coords, arr.dims, attrs=arr.attrs)
    if "id" in filtered_arr.attrs:
        del filtered_arr.attrs["id"]
        provenance_context: Provenance = {
            "what": "Gaussian filtered data",
            "by": "gaussian_filter_arr",
            "sigma": sigma,
            "use_pixel": use_pixel,
        }

        provenance(filtered_arr, arr, provenance_context)
    return filtered_arr


def boxcar_filter_arr(
    arr: xr.DataArray,
    size: dict[Hashable, float] | None = None,
    iteration_n: int = 1,
    default_size: int = 1,
    *,
    use_pixel: bool = False,
) -> xr.DataArray:
    """Coordinate aware `scipy.ndimage.uniform_filter`.

    Args:
        arr: ARPES data
        size: Kernel size, specified in terms of axis units (if use_pixel is False).
              An axis that is not specified will have a kernel width of `default_size` in
              index units.  If set 0 as the size, the kernel size is set to 1 in index units, which
              means no filtering.
        iteration_n: Repeats n times.
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

    if use_pixel:
        integered_size: dict[Hashable, int] = {k: int(v) for k, v in size.items()}
    else:
        integered_size = {
            k: int(v / (arr.coords[k][1] - arr.coords[k][0])) for k, v in size.items()
        }

    for dim in arr.dims:
        if dim not in integered_size or integered_size[str(dim)] == 0:
            integered_size[str(dim)] = default_size
    widths_pixel: tuple[int, ...] = tuple([integered_size[str(k)] for k in arr.dims])
    array_values: NDArray[np.float64] = np.nan_to_num(arr.values, nan=0.0, copy=True)

    for _ in range(iteration_n):
        array_values = ndimage.uniform_filter(
            input=array_values,
            size=widths_pixel,
        ).astype(np.float64)
    filtered_arr = arr.G.with_values(array_values)
    if "id" in arr.attrs:
        del filtered_arr.attrs["id"]
        provenance_context: Provenance = {
            "what": "Boxcar filtered data",
            "by": "boxcar_filter_arr",
            "size": size,
            "use_pixel": use_pixel,
        }

        provenance(filtered_arr, arr, provenance_context)
    return filtered_arr


@update_provenance("Savitzky Golay Filter")
def savitzky_golay_filter(  # noqa: PLR0913
    data: xr.DataArray,
    window_length: int = 3,
    polyorder: int = 2,
    deriv: int = 0,
    mode: Literal["mirror", "constant", "nearest", "wrap", "interp"] = "interp",
    cval: float = 0.0,
    dim: Hashable = "",
) -> xr.DataArray:
    """Implements a Savitzky Golay filter with given window size.

    This function is a thin wrapper of scipy.signal.savgol_filter

    Args:
        data (xr.DataArray): Input data.
        window_length: Number of points in the window that the filter uses locally (must be odd).
        polyorder: The polynomial order used in the convolution,
            and must be less than window_length.
        deriv: the order of the derivative to compute (default = 0 means only smoothing)
        mode (str): Mode for savgol_filter (default: "interp").
        cval (float): Constant value used if mode == 'constant'.
        dim (str): The dimension along which the filter is to be applied

    Returns:
        Smoothed data.
    """
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    axis = data.dims.index(dim) if dim else -1
    dim = dim if dim else data.dims[0]
    coords_diffs = np.diff(data.coords[dim])
    assert np.allclose(coords_diffs, coords_diffs[0], rtol=1e-5, atol=1e-6), (
        f"The coordinates must be equally spaced. Consider to use interpolation. f{coords_diffs}"
    )
    return data.G.with_values(
        savgol_filter(
            data,
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            mode=mode,
            cval=cval,
            axis=axis,
        ),
    )


def savgol_filter_multi(
    data: xr.DataArray,
    axis_params: dict[str, tuple[int, int]],
    deriv: int = 0,
    mode: Literal["mirror", "constant", "nearest", "wrap", "interp"] = "interp",
    cval: float = 0.0,
    **kwargs: float,
) -> xr.DataArray:
    """Apply Savitzky-Golay filter to an xarray.DataArray along multiple dimensions.

    Args:
        data (xr.DataArray): The input DataArray.
        axis_params (dict): Dictionary mapping axis names to dicts of filter parameters,
            e.g., {"time": (11, 3), ...}
            # the first item in tuple is window-length, the second is polyorder.
            # (1, 0) is no smoothing.
        mode (str): Mode for savgol_filter (default: "interp").
        deriv: the order of the derivative to compute (default = 0 means only smoothing)
        cval (float): Constant value used if mode == 'constant'.
        **kwargs: Additional keyword arguments passed to savgol_filter.

    Returns:
        xr.DataArray: The filtered DataArray.
    """
    filtered = data

    for axis, params in axis_params.items():
        if axis not in filtered.dims:
            msg = f"Axis '{axis}' not found in DataArray dimensions {filtered.dims}"
            raise ValueError(msg)
        axis_kwargs = dict(
            window_length=params[0],
            polyorder=params[1],
            mode=mode,
            cval=cval,
            deriv=deriv,
            **kwargs,
        )

        filtered = xr.apply_ufunc(
            savgol_filter,
            filtered,
            input_core_dims=[[axis]],
            output_core_dims=[[axis]],
            kwargs=dict(axis=-1, **axis_kwargs),
            vectorize=True,
            output_dtypes=[filtered.dtype],
        )
    return filtered
