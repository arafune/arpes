"""This package contains utilities related to taking more complicated shaped selections around data.

Currently it houses just utilities for forming disk and annular selections out of data.
"""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from arpes.debug import setup_logger

from . import normalize_to_spectrum

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from numpy.typing import NDArray

    from arpes._typing.base import DataType, ReduceMethod, XrTypes

__all__ = (
    "fat_sel",
    "ravel_from_mask",
    "select_around",
    "select_around_data",
    "select_disk",
    "select_disk_mask",
    "unravel_from_mask",
)


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)

UNSPECIFIED = 0.1
DEFAULT_RADII: dict[str, float] = {
    "kp": 0.02,
    "kx": 0.02,
    "ky": 0.02,
    "kz": 0.05,
    "phi": 0.02,
    "beta": 0.02,
    "theta": 0.02,
    "psi": 0.02,
    "eV": 0.05,
    "delay": 0.2,
    "T": 2,
    "temperature": 2,
}


def fat_sel(
    data: XrTypes,
    widths: dict[Hashable, float] | None = None,
    method: ReduceMethod = "mean",
    **kwargs: float,
) -> XrTypes:
    """Allows integrating a selection over a small region.

    The produced dataset will be normalized by dividing by the number
    of slices integrated over.

    This can be used to produce temporary datasets that have reduced
    uncorrelated noise.

    Args:
        data: The data to be selected from.
        widths: Override the widths for the slices. Reasonable defaults are used otherwise.
                Defaults to None.
        method: Method for ruducing the data. Defaults to "mean".
        kwargs: slice dict. The width can also be specified by like "eV_wdith=0.1".
            (Will be Deprecated)

    Returns:
        The data after selection.

    Note: The width must be specified by width.  Not kwargs.
    """
    logger.debug(f"widths: {widths}")
    logger.debug(f"kwargs: {kwargs}")
    if widths is None:
        widths = {}
    assert isinstance(widths, dict)
    default_widths = DEFAULT_RADII

    if data.S.angle_unit == "Degrees":
        default_widths["phi"] = 1.0
        default_widths["beta"] = 1.0
        default_widths["theta"] = 1.0
        default_widths["psi"] = 1.0

    extra_kwargs: dict[Hashable, float] = {k: v for k, v in kwargs.items() if k not in data.dims}
    logger.debug(f"extra_kwargs: {extra_kwargs}")
    slice_center: dict[str, float] = {k: v for k, v in kwargs.items() if k in data.dims}
    logger.debug(f"slice_center: {slice_center}")
    slice_widths: dict[str, float] = {
        k: widths.get(k, extra_kwargs.get(k + "_width", default_widths.get(k)))
        for k in slice_center
    }
    logger.debug(f"slice_widths: {slice_widths}")
    slices = {
        k: slice(v - slice_widths[k] / 2, v + slice_widths[k] / 2) for k, v in slice_center.items()
    }
    sliced = data.sel(slices)

    if not any(slice_center.keys()):
        msg = "The slice center is not spcefied."
        raise TypeError(msg)
    if method == "mean":
        normalized = sliced.mean(slices.keys(), keep_attrs=True)
    elif method == "sum":
        normalized = sliced.sum(slices.keys(), keep_attrs=True)
    else:
        msg = "Method should be either 'mean' or 'sum'."
        raise RuntimeError(msg)

    for k, v in slice_center.items():
        normalized.coords[k] = v
    return normalized


def select_around_data(
    data: xr.DataArray,
    points: Mapping[Hashable, xr.DataArray],
    radius: dict[Hashable, float] | float | None = None,  # radius={"phi": 0.005}
    *,
    mode: ReduceMethod = "sum",
) -> xr.DataArray:
    """Performs a binned selection around a point or points.

    Can be used to perform a selection along one axis as a function of another, integrating a
    region in the other dimensions.

    Example:
        As an example, suppose we have a dataset with dimensions ('eV', 'kp', 'T',)
        and we also by fitting determined the Fermi momentum as a function of T, kp_F('T'),
        stored in the dataset kFs. Then we could select momentum integrated EDCs in a small
        window around the fermi momentum for each temperature by using

        >>> edcs = full_data.S.select_around_data(points={'kp': kFs}, radius={'kp': 0.04})

        The resulting data will be EDCs for each T, in a region of radius 0.04 inverse angstroms
        around the Fermi momentum.

    Args:
        data: The data to be selected from.
        points: The set of points where the selection should be performed.
        radius: The radius of the selection in each coordinate. If dimensions are omitted, a
                standard sized selection will be made as a compromise.
        mode: How the reduction should be performed, one of "sum" or "mean". Defaults to "sum"

    Returns:
        The binned selection around the desired point or points.
    """
    assert mode in {"sum", "mean"}, "mode parameter should be either sum or mean."
    assert isinstance(points, dict | xr.Dataset)
    radius = radius or {}
    if isinstance(points, xr.Dataset):
        points = {k: points[k].item() for k in points.data_vars}
    assert isinstance(points, dict)
    radius = _radius(points, radius)
    logger.debug(f"radius: {radius}")

    assert isinstance(radius, dict)
    logger.debug(f"iter(points.values()): {iter(points.values())}")

    along_dims = next(iter(points.values())).dims
    selected_dims = list(points.keys())

    new_dim_order = [d for d in data.dims if d not in along_dims] + list(along_dims)

    data_for = data.transpose(*new_dim_order)
    new_data = data_for.sum(selected_dims, keep_attrs=True)

    stride: dict[Hashable, float] = data.G.stride(generic_dim_names=False)
    for coord in data_for.G.iter_coords(along_dims):
        value = data_for.sel(coord, method="nearest")
        nearest_sel_params: dict[Hashable, xr.DataArray] = {}
        for dim, v in radius.items():
            if v < stride[dim]:
                nearest_sel_params[dim] = points[dim].sel(coord)
        radius = {dim: v for dim, v in radius.items() if dim not in nearest_sel_params}
        selection_slices = {
            dim: slice(
                points[dim].sel(coord) - radius[dim],
                points[dim].sel(coord) + radius[dim],
            )
            for dim in points
            if dim in radius
        }
        selected = value.sel(selection_slices)
        if nearest_sel_params:
            selected = selected.sel(nearest_sel_params, method="nearest")
        for d in nearest_sel_params:
            del selected.coords[d]
        if mode == "sum":
            new_data.loc[coord] = selected.sum(list(radius.keys())).values
        elif mode == "mean":
            new_data.loc[coord] = selected.mean(list(radius.keys())).values
    return new_data


def select_around(
    data: xr.DataArray,
    point: dict[Hashable, float],
    radius: dict[Hashable, float] | float | None,
    *,
    mode: ReduceMethod = "sum",
) -> xr.DataArray:
    """Selects and integrates a region around a one dimensional point.

    This method is useful to do a small region integration, especially around
    point on a path of a k-point of interest. See also the companion method
    `select_around_data`.

    Args:
        data: The data to be selected from.
        point: The point where the selection should be performed.
        radius: The radius of the selection in each coordinate. If dimensions are omitted, a
                standard sized selection will be made as a compromise.
        safe: If true, infills radii with default values. Defaults to `True`.
        mode: How the reduction should be performed, one of "sum" or "mean". Defaults to "sum"

    Returns:
        The binned selection around the desired point.
    """
    assert mode in {"sum", "mean"}, "mode parameter should be either sum or mean."
    assert isinstance(point, dict | xr.Dataset)
    radius = _radius(point, radius)
    stride = data.G.stride(generic_dim_names=False)
    nearest_sel_params: dict[Hashable, float] = {}
    for dim, v in radius.items():
        if v < stride[dim]:
            nearest_sel_params[dim] = point[dim]
    radius = {dim: v for dim, v in radius.items() if dim not in nearest_sel_params}
    selection_slices = {
        dim: slice(point[dim] - radius[dim], point[dim] + radius[dim])
        for dim in point
        if dim in radius
    }
    selected = data.sel(selection_slices)
    if nearest_sel_params:
        selected = selected.sel(nearest_sel_params, method="nearest")
    for d in nearest_sel_params:
        del selected.coords[d]
    if mode == "sum":
        return selected.sum(list(radius.keys()))
    return selected.mean(list(radius.keys()))


def ravel_from_mask(data: DataType, mask: XrTypes) -> DataType:
    """Selects out the data from a NDArray whose points are marked true in `mask`.

    See also `unravel_from_mask`
    below which allows you to write back into data after you have transformed the 1D output in some
    way.

    These two functions are especially useful for hierarchical curve fitting where you want to rerun
    a fit over a subset of the data with a different model, such as when you know some of the data
    is best described by two bands rather than one.

    Args:
        data (DataType): Input ARPES data
        mask (XrTypes):  Mask data

    Returns:
        Raveled data with masked points removed.
    """
    return data.stack(stacked=list(mask.dims)).where(mask.stack(stacked=list(mask.dims)), drop=True)


def unravel_from_mask(
    template: xr.DataArray,
    mask: xr.DataArray,
    *,
    values: bool | float,
    default: float = np.nan,
) -> xr.DataArray:
    """Creates an array from a mask and a flat collection of the unmasked values.

    Inverse to `ravel_from_mask`, so look at that function as well.

    Args:
        template (DataType): Template for mask data
        mask (xr.DataArray | xr.Dataset): mask data
        values: default value
        default (float): [ToDo: search for what this "default" means]

    Returns:
        Unraveled data with default values filled in where the raveled list is missing from the mask
    """
    dest = template * 0 + 1
    dest_mask = np.logical_not(
        np.isnan(
            template.stack(stacked=list(template.dims))
            .where(mask.stack(stacked=list(template.dims)))
            .values,
        ),
    )
    dest = (dest * default).stack(stacked=list(template.dims))
    dest.values[dest_mask] = values
    return dest.unstack("stacked")


def select_disk_mask(
    data: xr.DataArray,
    radius: float,
    outer_radius: float | None = None,
    around: dict[str, float] | None = None,
    *,
    flat: bool = False,
) -> NDArray[np.float64]:
    """A complement to `select_disk` which only generates the mask for the selection.

    Selects the data in a disk around the point described by `around`. A point is a labelled
    collection of coordinates that matches all of the dimensions of `data`. The radius for the disk
    is specified through the required `radius` parameter.

    Returns the ND mask that represents the filtered coordinates.

    Args:
        data: The data which should be masked
        radius: The radius of the circle to mask
        outer_radius: The outer radius of an annulus to mask
        around: The location of the center point.
        flat: Whether to return the mask as a 1D (raveled) mask
          (flat=True) or as a ND mask with the same dimensionality as
          the input data (flat=False).

    Returns:
        A mask with the same shape as `data`.
    """
    data_array = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)

    raveled = data_array.G.ravel()
    assert around is not None

    dim_order = list(around.keys())
    dist = np.sqrt(
        np.sum(np.stack([(raveled[d] - around[d]) ** 2 for d in dim_order], axis=1), axis=1),
    )

    mask = dist <= radius
    if outer_radius is not None:
        mask = np.logical_or(mask, dist > outer_radius)

    if flat:
        return mask

    return mask.reshape(data_array.shape[::-1])


def select_disk(
    data: xr.DataArray,
    radius: float,
    outer_radius: float | None = None,
    around: dict[str, float] | None = None,
    *,
    invert: bool = False,
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.float64], NDArray[np.float64]]:
    """Selects the data in a disk around the point requested.

     (or annulus if `outer_radius` is provided)

    A point is a labeled collection of coordinates that matches all of the dimensions of `data`.
    The coordinates can be passed through a dict as `around`. The radius for the disk is
    specified through the required `radius` parameter.

    Data is returned as a tuple with the type tuple[dict[str, np.ndarray], np.ndarray,
    containing a dictionary with the filtered lists of coordinates, an array with the original data
    values at these coordinates, and finally an array of the distances to the requested point.

    Args:
        data: The data to perform the selection from
        radius: The inner radius of the annulus selection
        outer_radius: The outer radius of the annulus selection
        around: The central point.
        invert: Whether to invert the mask, i.e. everything but the annulus
    """
    data_array = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    mask = select_disk_mask(data_array, radius, outer_radius=outer_radius, around=around, flat=True)
    assert around is not None

    if invert:
        mask = np.logical_not(mask)

    # at this point, around is now a dictionary specifying a point to do the selection around
    raveled = data_array.G.ravel()

    dim_order = list(around.keys())
    dist = np.sqrt(
        np.sum(np.stack([(raveled[d] - around[d]) ** 2 for d in dim_order], axis=1), axis=1),
    )

    masked_coords = {d: cs[mask] for d, cs in raveled.items()}
    return masked_coords, masked_coords["data"], dist[mask]


def _radius(
    points: dict[Hashable, xr.DataArray] | dict[Hashable, float],
    radius: float | dict[Hashable, float] | None,
) -> dict[Hashable, float]:
    """Helper function. Generate radius dict.

    When radius is dict form, nothing has been done, essentially.

    Args:
        points (dict[Hashable, xr.DataArray] | dict[Hashable, float]): Selection point
        radius (dict[Hashable, float] | float | None): radius

    Returns: dict[Hashable, float]
        radius for selection.
    """
    if isinstance(radius, float):
        return dict.fromkeys(points, radius)
    if radius is None:
        radius = {d: DEFAULT_RADII.get(str(d), UNSPECIFIED) for d in points}
    if not isinstance(radius, dict):
        msg = "radius should be a float, dictionary or None"
        raise TypeError(msg)
    return radius
