"""This module contains methods that get unitful alignments of one array against another.

This is very useful for determining spectral shifts before doing serious curve fitting analysis or
similar.

Implementations are included for each of 1D and 2D arrays, but this could be simply extended to ND
if we need to. I doubt that this is necessary and don't mind the copied code too much at the
present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from lmfit.models import QuadraticModel
from scipy import signal

from arpes.constants import TWO_DIMENSION

if TYPE_CHECKING:
    import xarray as xr
    from numpy.typing import NDArray

__all__ = ("align",)


def align2d(a: xr.DataArray, b: xr.DataArray, *, subpixel: bool = True) -> tuple[float, float]:
    """Returns the unitful offset of b in a for 2D arrays using 2D correlation.

    Args:
        a: The first input array.
        b: The second input array.
        subpixel(bool): If True, will perform subpixel alignment using a curve fit to the peak.

    Returns:
        The offset of a 2D array against another.
    """
    corr = signal.correlate2d(
        a.values - np.mean(a.values),
        b.values - np.mean(b.values),
        boundary="fill",
        mode="same",
    )
    y, x = np.unravel_index(np.argmax(corr), corr.shape)

    if subpixel:
        y_offset = _quadratic_fit(corr[y - 10 : y + 10, x])
        x_offset = _quadratic_fit(corr[y, x - 10 : x + 10])
    else:
        y_offset = 0
        x_offset = 0

    return (
        (float(y + y_offset) - (a.values.shape[0] - 1) // 2)
        * a.G.stride(generic_dim_names=False)[a.dims[0]],
        (float(x + x_offset) - (a.values.shape[1] - 1) // 2)
        * a.G.stride(generic_dim_names=False)[a.dims[1]],
    )


def align1d(a: xr.DataArray, b: xr.DataArray, *, subpixel: bool = True) -> float:
    """Returns the unitful offset of b in a for 1D arrays using 1D correlation.

    Args:
        a: The first input array.
        b: The second input array.
        subpixel: If True, will perform subpixel alignment using a curve fit to the peak.

    Returns:
        The offset of an array against another.
    """
    corr = np.correlate(a.values - np.mean(a.values), b.values - np.mean(b.values), mode="same")
    (x,) = np.unravel_index(np.argmax(corr), corr.shape)

    x_offset = _quadratic_fit(corr[x - 10 : x + 10]) if subpixel else 0.0
    return (float(x + x_offset) - (a.values.shape[0]) // 2) * a.G.stride(
        generic_dim_names=False,
    )[a.dims[0]]


def align(a: xr.DataArray, b: xr.DataArray, **kwargs: bool) -> tuple[float, float] | float:
    """Returns the unitful offset of b in a for ndarrays.

    Args:
        a(xr.DataArray): The first input array.
        b(xr.DataArray): The second input array.
        kwargs(bool): Pass to align2d (and currently only subpixel = True/False is accepted)

    Returns:
        The offset of an array against another.
    """
    if len(a.dims) == 1:
        return align1d(a, b, **kwargs)

    assert len(a.dims) == TWO_DIMENSION
    return align2d(a, b, **kwargs)


def _quadratic_fit(peak_neigbor: NDArray[np.float64]) -> float:
    """Helper function for QuadraticModel fit to determie the peak location.

    Args:
        peak_neigbor (NDArray[np.float64]): 1D NDarary around the peak. The length must be 20.

    Returns:
        float: correction of peak position with subpixel resolution.
    """
    model = QuadraticModel()
    x = np.linspace(-10, 9, 20)
    params = model.guess(peak_neigbor / np.max(peak_neigbor), x)
    result = model.fit(data=peak_neigbor / np.max(peak_neigbor), params=params, x=x)
    return -result.params["b"].value / (2 * result.params["a"].value)
