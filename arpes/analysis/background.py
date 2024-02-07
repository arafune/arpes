"""Provides background estimation approaches."""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull

if TYPE_CHECKING:
    from numpy._typing import NDArray

__all__ = (
    "calculate_background_hull",
    "remove_background_hull",
)


def calculate_background_hull(
    arr: xr.DataArray,
    breakpoints: list[float | None] | None = None,
) -> xr.DataArray:
    """Calculates background using the convex hull of the data (intensity as a Z axis)."""
    assert isinstance(arr, xr.DataArray)
    assert len(arr.dims) == 1
    if breakpoints:
        breakpoints = [None, *breakpoints, None]
        dim = arr.dims[0]
        processed = []
        for blow, bhigh in pairwise(breakpoints):
            processed.append(calculate_background_hull(arr.sel({dim: slice(blow, bhigh)})))
        return xr.concat(processed, dim)

    points = np.stack(arr.G.to_arrays(), axis=1)
    hull = ConvexHull(points)

    vertices: NDArray[np.float_] = np.array(hull.vertices)
    index_of_zero = np.argwhere(vertices == 0)[0][0]
    vertices = np.roll(vertices, -index_of_zero)
    xis = [*list(vertices[: np.argwhere(vertices == len(arr) - 1)[0][0]]), len(arr) - 1]

    support = points[xis]
    interp1d(support[:, 0], support[:, 1], fill_value="extrapolate")(points[:, 0])
    return arr.S.with_values(interp1d(support[:, 0], support[:, 1])(points[:, 0]))


def remove_background_hull(
    data: xr.DataArray,
    *args: list[float | None] | None,
    **kwargs: list[float | None] | None,
) -> xr.DataArray:
    """Removes a background according to `calculate_background_hull`."""
    return data - calculate_background_hull(data, *args, **kwargs)
