"""Low-level coordinate transformation utilities.

This module contains internal, low-level routines used to convert ARPES data
between coordinate systems (e.g. angle space to momentum space) using
volumetric interpolation.

⚠️ Design notes
----------------
The functions defined here are **NOT part of the public API**.

They intentionally expose implementation details such as:
- explicit target coordinate grids,
- transformation dictionaries,
- interpolation order and reshaping rules,
- assumptions about xarray dimensions and accessors.

As a result:
- This module MUST NOT be re-exported from ``arpes.utilities.conversion.__init__``.
- Users should NOT import from this module directly.
- Backward compatibility is NOT guaranteed.

Public-facing coordinate conversion functionality is provided instead via:
- higher-level analysis routines (e.g. ``convert_to_kspace``),
- xarray accessors (``DataArray.S``),
- or dedicated user APIs in the ``analysis`` layer.

🧱 Architectural role
---------------------
This module belongs to the **internal utility layer** and is designed to be:

- shared by plotting and analysis code,
- flexible and powerful,
- free to evolve without API constraints.

It should depend only on:
- NumPy,
- xarray,
- and other internal utilities,

and must not introduce dependencies on higher-level modules such as
``analysis``, ``plotting``, or xarray accessors beyond what is strictly required.

If you are looking for a user-facing function, this is probably not the module
you want to import.
"""

from __future__ import annotations

import contextlib
from logging import DEBUG, INFO
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import xarray as xr

from arpes.debug import setup_logger

from .core import grid_interpolator_from_dataarray
from .fast_interp import Interpolator

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from numpy.typing import NDArray

    from arpes._typing.base import XrTypes


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


class CoordinateTransform(TypedDict, total=True):
    """Internal specification of a coordinate transformation.

    This TypedDict defines the minimal contract required by the low-level
    volumetric coordinate conversion engine. It describes:

    - the ordered set of target dimensions in the output space, and
    - a mapping from each source/target dimension name to a callable
      that computes the corresponding coordinate values.

    ⚠️ This is an internal data structure.
    ------------------------------------
    It is **not part of the public API** and may change without notice.
    Users should not construct or rely on this object directly.

    The structure is intentionally lightweight and flexible to support
    different coordinate systems (e.g. angle space, momentum space) without
    imposing a rigid class hierarchy.

    Fields
    ------
    dims : list[str] or list[Hashable]
        Ordered names of the target coordinate dimensions.
        The order determines the shape and ordering of the output array.

        In most practical ARPES use cases, this will be something like::

            ["kp"]                    # cut data
            ["kx", "ky"]              # 2D momentum maps
            ["kx", "ky", "kz"]         # 3D momentum volumes

        but no specific coordinate system is assumed at this level.

    transforms : dict[str, Callable[..., NDArray[np.floating]]]
        Mapping from coordinate names to transformation functions.

        Each callable must accept a sequence of meshed coordinate arrays
        (as produced by ``numpy.meshgrid``) and return a NumPy array of
        transformed coordinate values.

        The keys of this dictionary are expected to include:
        - all original dimensions of the input DataArray, and
        - all target dimensions listed in ``dims``.

    Notes:
    - No validation of physical correctness is performed here.
    - The numerical meaning of the transforms is defined entirely by
      the calling code.
    - This structure is designed to support volumetric interpolation
      workflows and should remain free of higher-level concepts such as
      spectrum type, plotting logic, or experiment metadata.
    """

    dims: list[str] | list[Hashable]  # in most case dims should be Literal["kp", "kx", "ky", "kz"]]
    transforms: dict[str, Callable[..., NDArray[np.floating]]]


def convert_coordinates(
    arr: xr.DataArray,
    target_coordinates: dict[Hashable, NDArray[np.floating]],
    coordinate_transform: CoordinateTransform,
    *,
    as_dataset: bool = False,
) -> XrTypes:
    """Return the band structure data (converted to k-space).

    Args:
        arr(xr.DataArray): ARPES data
        target_coordinates:(dict[Hashable, NDArray[np.floating]]):  coorrdinate for ...
        coordinate_transform(dict[str, list[str] | Callable]): coordinat for ...
        as_dataset(bool): if True, return the data as the dataSet

    Returns:
        XrTypes
    """
    assert isinstance(arr, xr.DataArray)
    ordered_source_dimensions = arr.dims

    grid_interpolator = grid_interpolator_from_dataarray(
        arr.transpose(*ordered_source_dimensions),  # TODO(RA): No need? -- perhaps no.
        fill_value=np.nan,
    )

    # Skip the Jacobian correction for now
    # Convert the raw coordinate axes to a set of gridded points
    logger.debug(
        f"meshgrid: {[len(target_coordinates[dim]) for dim in coordinate_transform['dims']]}",
    )
    meshed_coordinates = [
        meshed_coord.ravel()
        for meshed_coord in np.meshgrid(
            *[target_coordinates[dim] for dim in coordinate_transform["dims"]],
            indexing="ij",
        )
    ]

    if "eV" not in arr.dims:
        with contextlib.suppress(ValueError):
            meshed_coordinates = [arr.S.lookup_offset_coord("eV"), *meshed_coordinates]
    old_coord_names = [str(dim) for dim in arr.dims if dim not in target_coordinates]
    assert isinstance(coordinate_transform["transforms"], dict)
    transforms: dict[str, Callable[..., NDArray[np.floating]]] = coordinate_transform["transforms"]
    logger.debug(f"transforms is {transforms}")
    old_coordinate_transforms = [
        transforms[str(dim)] for dim in arr.dims if dim not in target_coordinates
    ]
    logger.debug(f"old_coordinate_transforms: {old_coordinate_transforms}")

    output_shape = [len(target_coordinates[str(d)]) for d in coordinate_transform["dims"]]

    def compute_coordinate(transform: Callable[..., NDArray[np.floating]]) -> NDArray[np.floating]:
        logger.debug(f"transform function is {transform}")
        return np.reshape(
            transform(*meshed_coordinates),
            output_shape,
            order="C",
        )

    old_dimensions = [compute_coordinate(tr) for tr in old_coordinate_transforms]

    ordered_transformations = [transforms[str(dim)] for dim in arr.dims]
    transformed_coordinates = [tr(*meshed_coordinates) for tr in ordered_transformations]

    converted_volume = (
        grid_interpolator(np.array(transformed_coordinates).T)
        if not isinstance(grid_interpolator, Interpolator)
        else grid_interpolator(transformed_coordinates)
    )

    # Wrap it all up
    def acceptable_coordinate(c: NDArray[np.floating] | xr.DataArray) -> bool:
        """Return True if the dim of array is subset of dim of coordinate_transform.

        Currently we do this to filter out coordinates
        that are functions of the old angular dimensions,
        we could forward convert these, but right now we do not

        Args:
            c (xr.DataArray): DataArray for check.

        Returns: bool
            Return True if the dim of array is subset of dim of coordinate_transform.
        """
        if isinstance(c, xr.DataArray):
            return set(c.dims).issubset(coordinate_transform["dims"])
        return True

    target_coordinates = {k: v for k, v in target_coordinates.items() if acceptable_coordinate(v)}
    data = xr.DataArray(
        np.reshape(
            converted_volume,
            [len(target_coordinates[str(d)]) for d in coordinate_transform["dims"]],
            order="C",
        ),
        target_coordinates,
        coordinate_transform["dims"],
        attrs=arr.attrs,
    )
    if as_dataset:
        old_mapped_coords = [
            xr.DataArray(
                values,
                coords=target_coordinates,
                dims=coordinate_transform["dims"],
                attrs=arr.attrs,
            )
            for values in old_dimensions
        ]
        variables = {"data": data}
        variables.update(
            dict(
                zip(
                    old_coord_names,
                    old_mapped_coords,
                    strict=True,
                ),
            ),
        )
        return xr.Dataset(variables, attrs=arr.attrs)

    return data
