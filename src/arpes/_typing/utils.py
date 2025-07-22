"""Specialized type annotations for use in PyARPES.

In particular, `DataType` refers to either an xarray.DataArray or xarray.Dataset

`NormalizableDataType` referes to anything that can be tuned into datase,
such as by loading from the cache using an ID.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    TypeGuard,
    get_args,
)

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Hashable, Sequence

    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from .attrs_property import KspaceCoords
    from .base import DataType, XrTypes


def flatten_literals(literal_type: Incomplete) -> set[str]:
    """Recursively flattens a Literal type to extract all string values.

    Args:
        literal_type (type[Literal] | Literal): The Literal type to flatten.

    Returns:
        set[str]: A set of all string values in the Literal type.
    """
    args = get_args(literal_type)
    flattened = set()
    for arg in args:
        if hasattr(arg, "__args__"):
            flattened.update(flatten_literals(arg))
        else:
            flattened.add(arg)
    return flattened


def is_dict_kspacecoords(
    a_dict: dict[Hashable, NDArray[np.float64]] | dict[str, NDArray[np.float64]],
) -> TypeGuard[KspaceCoords]:
    """Checks if a dictionary contains k-space coordinates.

    Args:
        a_dict (dict[Hashable, NDArray[np.float64]] | dict[str, NDArray[np.float64]]):
           The dictionary to check.

    Returns:
        TypeGuard[KspaceCoords]: True if the dictionary contains k-space coordinates,
        False otherwise.
    """
    if not a_dict:
        return False
    return all(
        key in {"eV", "kp", "kx", "ky", "kz"} and isinstance(a_dict[str(key)], np.ndarray)
        for key in a_dict
    )


def is_homogeneous_dataarray_list(
    arr_list: Sequence[XrTypes] | Sequence[DataType],
) -> TypeGuard[Sequence[xr.DataArray]]:
    """Check if all elemetns in the list are of type xr.DataArray."""
    return all(isinstance(arr, xr.DataArray) for arr in arr_list)


def is_homogeneous_dataset_list(
    arr_list: Sequence[XrTypes] | Sequence[DataType],
) -> TypeGuard[Sequence[xr.Dataset]]:
    """Check if all elemetns in the list are of type xr.Dataset."""
    return all(isinstance(arr, xr.Dataset) for arr in arr_list)


def is_dims_match_coordinate_convert(
    angles: tuple[str, ...],
) -> TypeGuard[
    tuple[Literal["phi"]]
    | tuple[Literal["beta"], Literal["phi"]]
    | tuple[Literal["phi"], Literal["theta"]]
    | tuple[Literal["phi"], Literal["psi"]]
    | tuple[Literal["hv"], Literal["phi"]]
    | tuple[Literal["beta"], Literal["hv"], Literal["phi"]]
    | tuple[Literal["hv"], Literal["phi"], Literal["theta"]]
    | tuple[Literal["hv"], Literal["phi"], Literal["psi"]]
]:
    return angles in {
        ("phi",),
        ("theta",),
        ("beta",),
        ("phi", "theta"),
        ("phi", "psi"),
        ("beta", "phi"),
        ("hv", "phi"),
        ("hv",),
        ("beta", "hv", "phi"),
        ("hv", "phi", "theta"),
        ("hv", "phi", "psi"),
        ("chi", "hv", "phi"),
    }
