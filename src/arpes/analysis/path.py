"""Contains routines used to do path selections and manipulations on a dataset."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from arpes.provenance import update_provenance

if TYPE_CHECKING:
    from _typeshed import Incomplete
    from numpy.typing import NDArray


__all__ = (
    "discretize_path",
    "select_along_path",
)


@update_provenance("Discretize Path")
def discretize_path(
    path: xr.Dataset,
    n_points: int = 0,
    scaling: float | xr.Dataset | dict[str, NDArray[np.float_]] | None = None,
) -> xr.Dataset:
    """Discretizes a path into a set of points spaced along the path.

    Shares logic with slice_along_path

    Args:
        path: The path specification.
        n_points: The number of points to space along the path.
        scaling: A metric allowing calculating a distance from mixed coordinates. This
          is needed because we disperse points equidistantly along the path. Typically
          you can leave this unset.

    Returns:
        An xr.Dataset of the points along the path.
    """
    if scaling is None:
        scaling = 1
    elif isinstance(scaling, xr.Dataset):
        scaling = {str(k): scaling[k].item() for k in scaling.data_vars}
    else:
        assert isinstance(scaling, dict)

    order = list(path.data_vars)
    if isinstance(scaling, dict):
        scaling = np.array(scaling[d] for d in order)

    def as_vec(ds: xr.Dataset) -> NDArray[np.float_]:
        return np.array([ds[k].item() for k in order])

    def distance(a: xr.Dataset, b: xr.Dataset) -> float:
        return np.linalg.norm((as_vec(a) - as_vec(b)) * scaling)

    length = 0
    for idx_low, idx_high in zip(path.index.values, path.index[1:].values, strict=False):
        coord_low, coord_high = path.sel(index=idx_low), path.sel(index=idx_high)
        length += distance(coord_low, coord_high)

    n_points = int(length / 0.03) if not n_points else max(n_points - 1, 1)

    sep = length / n_points
    points = []
    distances = np.linspace(0, n_points - 1, n_points) * sep

    total_dist = 0
    for idx_low, idx_high in zip(path.index.values, path.index[1:].values, strict=False):
        coord_low, coord_high = path.sel(index=idx_low), path.sel(index=idx_high)

        current_dist = distance(coord_low, coord_high)
        current_points = distances[distances < total_dist + current_dist]
        current_points = (current_points - total_dist) / current_dist
        distances = distances[len(current_points) :]
        total_dist += current_dist

        points = points + list(
            np.outer(current_points, as_vec(coord_high) - as_vec(coord_low)) + as_vec(coord_low),
        )

    points.append(as_vec(path.sel(index=path.index.values[-1])))

    new_index = np.array(range(len(points)))

    def to_dataarray(name: str) -> xr.DataArray:
        index = order.index(name)
        data = [p[index] for p in points]

        return xr.DataArray(np.array(data), {"index": new_index}, ["index"])

    return xr.Dataset({k: to_dataarray(k) for k in order})


@update_provenance("Select from data along a path")
def select_along_path(
    path: xr.Dataset,
    data: xr.DataArray,
    radius: float = 0,
    n_points: int = 0,
    *,
    scaling: float | xr.Dataset | dict[str, NDArray[np.float_]] | None = None,
    **kwargs: Incomplete,
) -> xr.DataArray:
    """Performs integration along a path.

    This functionally allows for performing a finite width
    cut (with finite width perpendicular to the local path direction) along some path,
    and integrating along this perpendicular selection. This allows for better statistics in
    oversampled data.

    Args:
        path: The path to select along.
        data: The data to select/interpolate from.
        radius: A number or dictionary of radii to use for the selection along different dimensions,
                if none is provided reasonable values will be chosen. Alternatively, you can pass
                radii via `{dim}_r` kwargs as well, i.e. 'eV_r' or 'kp_r'
        n_points: The number of points to interpolate along the path, by default we will infer a
                  reasonable number from the radius parameter, if provided or inferred
        scaling: A metric allowing calculating a distance from mixed coordinates.
                 Pass it to discretize_path as is.
        kwargs: kwargs pass to S.select_around

    Returns:
        The data selected along the path.
    """
    new_path = discretize_path(path, n_points, scaling)

    selections = []
    for _, view in new_path.G.iterate_axis("index"):
        selections.append(data.S.select_around(view, radius=radius, **kwargs))

    return xr.concat(selections, new_path.index)
