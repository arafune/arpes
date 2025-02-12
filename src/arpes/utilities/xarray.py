"""Utilities related to function application on xr types."""

from __future__ import annotations

import warnings
from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar

import xarray as xr

from arpes.debug import setup_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes._typing import DataType

__all__ = (
    "lift_dataarray",
    "lift_dataarray_attrs",
    "lift_datavar_attrs",
    "unwrap_xarray_dict",
    "unwrap_xarray_item",
)


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)

T = TypeVar("T")


def unwrap_xarray_item(item: xr.DataArray) -> xr.DataArray | float:
    """Unwraps xr.DataArray that might or might not be an xarray like with .item() attribute.

    This is especially helpful for dealing with unwrapping coordinates which might
    be floating point-like or might be array-like.

    Args:
        item: The value to unwrap.

    Returns:
        The safely unwrapped item
    """
    warnings.warn(
        "This method will be deprecated. (unwrap_xarray_item)",
        category=PendingDeprecationWarning,
        stacklevel=2,
    )

    try:
        return item.item()
    except (AttributeError, ValueError):
        return item


def unwrap_xarray_dict(
    input_dict: dict[str, xr.DataArray],
) -> dict[str, xr.DataArray | float]:
    """Returns the attributes as unwrapped values rather than item() instances.

    Useful for unwrapping coordinate dicts where the values might be a bare type:
    like a float or an int, but might also be a wrapped array-like for instance
    xr.DataArray. Even worse, we might have wrapped bare values!

    Args:
        input_dict (dict[str, xr.DataArray]): input dictionary

    Returns:
        The unwrapped attributes as a dict.

    """
    warnings.warn(
        "This method will be deprecated. (unwrap_xarray_dict)",
        category=PendingDeprecationWarning,
        stacklevel=2,
    )

    return {k: unwrap_xarray_item(v) for k, v in input_dict.items()}


def apply_dataarray(
    arr: xr.DataArray,  # arr.values is used
    f: Callable[[NDArray[np.float64], Any], NDArray[np.float64]],
    *args: Incomplete,
    **kwargs: Incomplete,
) -> xr.DataArray:
    """Applies a function onto the values of a DataArray.

    Args:
        arr (xr.DataArray): original DataArray.
        f (Callable): Function to apply the DataArray.
        args: arguments for "f".
        kwargs: keyword arguments for "f"

    Returns:
        xr.DataArray replaced after the function.
    """
    return arr.G.with_values(f(arr.values, *args, **kwargs))


def lift_dataarray(  # unused
    f: Callable[[NDArray[np.float64], Any], NDArray[np.float64]],
) -> Callable[[xr.DataArray], xr.DataArray]:
    """Lifts a function that operates on an np.ndarray's values to act on an xr.DataArray.

    Args:
        f: Callable

    Returns:
        g: Function operating on an xr.DataArray
    """

    def g(arr: xr.DataArray, *args: Incomplete, **kwargs: Incomplete) -> xr.DataArray:
        return apply_dataarray(arr, f, *args, **kwargs)

    return g


P = ParamSpec("P")


def lift_dataarray_attrs(
    f: Callable[Concatenate[dict[str, T], P], dict[str, T]],
) -> Callable[Concatenate[xr.DataArray, P], xr.DataArray]:
    """Lifts a function that operates dicts to a function that acts on dataarray attrs.

    Produces a new xr.DataArray.

    Args:
        f: Function to apply

    Returns:
        g: Function operating on the attributes of an xr.DataArray
    """

    def g(
        arr: xr.DataArray,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> xr.DataArray:
        """Applies the function to the attributes of the dataarray.

        Args:
            arr (xr.DataArray): ARPES data
            *args: Additional arguments to pass to the function "f".
            **kwargs: Additional keyword arguments to pass to the function "f".

        Returns:
            xr.DataArray: New DataArray with modified attributes.
        """
        return xr.DataArray(
            arr.values,
            arr.coords,
            arr.dims,
            attrs=f(arr.attrs, *args, **kwargs),
        )

    return g


def lift_datavar_attrs(
    f: Callable[Concatenate[dict[str, T], P], dict[str, T]],
) -> Callable[Concatenate[DataType, P], DataType]:
    """Lifts a function that operates dicts to a function that acts on xr attrs.

    Applies to all attributes of all the datavars in a xr.Dataset, as well as the Dataset
    attrs themselves.

    Args:
        f: Function to apply

    Returns:
        The function modified to apply to xr instances.
    """

    def g(
        data: DataType,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> DataType:
        """Applies the function to the attributes of the data.

        Args:
            data (DataType): ARPES Data (Dataset or DataArray) with attributes to modify.
            *args: Additional arguments to pass to the function "f"
            **kwargs: Additional keyword arguments to pass to the function "f"
        """
        arr_lifted = lift_dataarray_attrs(f)
        if isinstance(data, xr.DataArray):
            return arr_lifted(data, *args, **kwargs)

        new_vars = {k: arr_lifted(da, *args, **kwargs) for k, da in data.data_vars.items()}
        new_root_attrs = f(data.attrs, *args, **kwargs)

        return xr.Dataset(new_vars, data.coords, new_root_attrs)

    return g
