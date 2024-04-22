"""This package contains utilities related to taking more complicated shaped selections around data.

Currently it houses just utilities for forming disk and annular selections out of data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from . import normalize_to_spectrum
from .xarray import unwrap_xarray_dict

if TYPE_CHECKING:
    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes._typing import DataType, XrTypes

__all__ = ("select_disk", "select_disk_mask", "unravel_from_mask", "ravel_from_mask")


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
    template: DataType,
    mask: XrTypes,
    values: bool | float,
    default: float = np.nan,
) -> DataType:
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


def _normalize_point(
    data: xr.DataArray,
    around: dict[str, xr.DataArray] | xr.Dataset | None,
    **kwargs: NDArray[np.float_] | float,
) -> dict[str, xr.DataArray]:
    collected_kwargs = {k: kwargs[str(k)] for k in data.dims if k in kwargs}

    if around:
        if isinstance(around, xr.Dataset):
            around = unwrap_xarray_dict({str(d): around[d] for d in data.dims})
    else:
        around = collected_kwargs

    assert isinstance(around, dict)
    assert set(around.keys()) == set(data.dims)
    return around


def select_disk_mask(
    data: xr.DataArray,
    radius: float,
    outer_radius: float | None = None,
    around: dict | xr.Dataset | None = None,
    *,
    flat: bool = False,
    **kwargs: Incomplete,
) -> NDArray[np.float_]:
    """A complement to `select_disk` which only generates the mask for the selection.

    Selects the data in a disk around the point described by `around` and `kwargs`. A point is a
    labelled collection of coordinates that matches all of the dimensions of `data`. The
    coordinates can either be passed through a dict as `around`, as the coordinates of a Dataset
    through `around` or explicitly in keyword argument syntax through `kwargs`. The radius for the
    disk is specified through the required `radius` parameter.

    Returns the ND mask that represents the filtered coordinates.

    Args:
        data: The data which should be masked
        radius: The radius of the circle to mask
        outer_radius: The outer radius of an annulus to mask
        around: The location of the center point otherwise specified by `kwargs`
        flat: Whether to return the mask as a 1D (raveled) mask
          (flat=True) or as a ND mask with the same dimensionality as
          the input data (flat=False).
        kwargs: The location of the center point otherwise specified by `around`

    Returns:
        A mask with the same shape as `data`.
    """
    if outer_radius is not None and radius > outer_radius:
        radius, outer_radius = outer_radius, radius

    data_array = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)

    around = _normalize_point(data_array, around, **kwargs)

    raveled = data_array.G.ravel()

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
    around: dict | xr.Dataset | None = None,
    *,
    invert: bool = False,
    **kwargs: Incomplete,
) -> tuple[dict[str, NDArray[np.float_]], NDArray[np.float_], NDArray[np.float_]]:
    """Selects the data in a disk around the point requested.

     (or annulus if `outer_radius` is provided)

    The point is specified by `around` and `kwargs`. A point is a labeled collection of coordinates
    that matches all of the dimensions of `data`. The coordinates can either be passed through a
    dict as `around`, as the coordinates of a Dataset through `around` or explicitly in keyword
    argument syntax through `kwargs`. The radius for the disk is specified through the required
    `radius` parameter.

    Data is returned as a tuple with the type tuple[dict[str, np.ndarray], np.ndarray,
    containing a dictionary with the filtered lists of coordinates, an array with the original data
    values at these coordinates, and finally an array of the distances to the requested point.

    Args:
        data: The data to perform the selection from
        radius: The inner radius of the annulus selection
        outer_radius: The outer radius of the annulus selection
        around: The central point, otherwise specified by `kwargs`
        invert: Whether to invert the mask, i.e. everything but the annulus
        kwargs: The central point, otherwise specified by `around`
    """
    data_array = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    around = _normalize_point(data_array, around, **kwargs)
    mask = select_disk_mask(data_array, radius, outer_radius=outer_radius, around=around, flat=True)

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
