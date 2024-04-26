"""Utilities for applying masks to data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from matplotlib.path import Path
from numpy.typing import NDArray

from arpes.provenance import update_provenance
from arpes.utilities import normalize_to_spectrum

if TYPE_CHECKING:
    from collections.abc import Iterable

    from _typeshed import Incomplete


__all__ = (
    "apply_mask",
    "apply_mask_to_coords",
    "polys_to_mask",
    "raw_poly_to_mask",
)


def raw_poly_to_mask(poly: Incomplete) -> dict[str, Incomplete]:
    """Converts a polygon into a mask definition.

    There's not currently much metadata attached to masks, but this is
    around if we ever decide that we need to implement more
    complicated masking schemes.

    In particular, we might want to store also whether the interior
    or exterior is the masked region, but this is functionally achieved
    for now with the `invert` flag in other functions.

    Args:
        poly: Polygon implementing a masked region.

    Returns:
        The mask.
    """
    return {
        "poly": poly,
    }


def polys_to_mask(
    mask_dict: dict[str, Incomplete],
    coords: xr.Coordinates,
    shape: list[tuple[int, ...]],
    radius: float = 0,
    *,
    invert: bool = False,
) -> NDArray[np.bool_]:
    """Converts a mask definition in terms of the underlying polygon to a True/False mask array.

    Uses the coordinates and shape of the target data in order to determine which pixels
    should be masked.

    This process "specializes" a mask to a particular shape, whereas masks given by
    polygon definitions are general to any data with appropriate dimensions, because
    waypoints are given in unitful values rather than index values.

    Args:
        mask_dict (dict): dict object to represent mask.
            dim and polys keys are required.
        coords (xr.coordinates): coordinates
        shape (list[tuple[int, ...]]):  Shape of mask
        radius (float): Additional margin on the path in coordinates of *points*.
        invert (bool): if true, flip True/False in mask.

    Returns:
        The mask.
    """
    dims = mask_dict["dims"]
    polys = mask_dict["polys"]

    polys = [
        [[np.searchsorted(coords[dims[i]], coord) for i, coord in enumerate(p)] for p in poly]
        for poly in polys
    ]

    mask_grids = np.meshgrid(*[np.arange(s) for s in shape])
    mask_grids = tuple(k.flatten() for k in mask_grids)

    points = np.vstack(mask_grids).T

    mask = None
    for poly in polys:
        grid: NDArray[np.bool_] = Path(poly).contains_points(points, radius=radius)

        grid = grid.reshape(list(shape)[::-1]).T

        mask = grid if mask is None else np.logical_or(mask, grid)
    assert isinstance(mask, NDArray[np.bool_])

    if invert:
        mask = np.logical_not(mask)

    return mask


def apply_mask_to_coords(
    data: xr.Dataset,  # data.data_vars is used
    mask: dict[str, NDArray[np.float_] | Iterable[Iterable[float]]],  # (N, 2) array
    dims: list[str],
    *,
    invert: bool = True,
) -> NDArray[np.bool_]:
    """Performs broadcasted masking along a given dimension.

    Args:
        data: The data you want to mask.
        mask: The mask to apply, should be dimensionally equivalent to what you request in `dims`.
        dims: The dimensions which should be masked.
        invert: Whether the mask should be inverted.

    Returns:
        The masked data.
    """
    as_array = np.stack([data.data_vars[d].values for d in dims], axis=-1)
    shape = as_array.shape
    dest_shape = shape[:-1]
    new_shape = [np.prod(dest_shape), len(dims)]
    mask_array = (
        Path(np.array(mask["poly"]))
        .contains_points(as_array.reshape(new_shape))
        .reshape(dest_shape)
    )

    if invert:
        mask_array = np.logical_not(mask_array)

    return mask_array


@update_provenance("Apply boolean mask to data")
def apply_mask(
    data: xr.DataArray,
    mask: dict[str, Incomplete] | NDArray[np.bool_],
    replace: float = np.nan,
    radius: Incomplete = None,
    *,
    invert: bool = False,
) -> xr.DataArray:
    """Applies a logical mask, i.e. one given in terms of polygons, to a specific piece of data.

    This can be used to set values outside or inside a series of
    polygon masks to a given value or to NaN.

    Expanding or contracting the masked region can be accomplished with the
    radius argument, but by default strict inclusion is used.

    Some masks include a `fermi` parameter which allows for clipping the detector
    boundaries in a semi-automated fashion. If this is included, only 200meV above the Fermi
    level will be included in the returned data. This helps to prevent very large
    and undesirable regions filled with only the replacement value which can complicate
    automated analyses that rely on masking.

    Args:
        data: Data to mask.
        mask: Logical definition of the mask, appropriate for passing to
            `polys_to_mask`
        replace: The value to substitute for pixels masked.
        radius: Radius by which to expand the masked area.
        invert: Allows logical inversion of the masked parts of the
            data. By default, the area inside the polygon sequence is
            replaced by `replace`.

    Returns:
        Data with values masked out.
    """
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    fermi: float | None = None

    if isinstance(mask, dict):
        fermi = mask.get("fermi", None)
        dims = mask.get("dims", data.dims)
        assert isinstance(mask, dict)
        mask_arr: NDArray[np.bool_] = polys_to_mask(
            mask,
            data.coords,
            [s for i, s in enumerate(data.shape) if data.dims[i] in dims],
            radius=radius,
            invert=invert,
        )
    else:
        mask_arr = mask

    masked_data = data.copy(deep=True)
    masked_data.values = masked_data.values * 1.0
    masked_data.values[mask_arr] = replace

    if fermi is not None:
        return masked_data.sel(eV=slice(None, fermi + 0.2))

    return masked_data
