"""Utilities related to function application on xr types."""

from __future__ import annotations

from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, Any, TypeVar

import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from _typeshed import Incomplete
    from numpy.typing import NDArray

    from arpes._typing import DataType

__all__ = (
    "apply_dataarray",
    "lift_datavar_attrs",
    "lift_dataarray_attrs",
    "lift_dataarray",
    "unwrap_xarray_item",
    "unwrap_xarray_dict",
)


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

T = TypeVar("T")


def unwrap_xarray_item(item: xr.DataArray) -> xr.DataArray | float:
    """Unwraps something that might or might not be an xarray like with .item() attribute.

    This is especially helpful for dealing with unwrapping coordinates which might
    be floating point-like or might be array-like.

    Args:
        item: The value to unwrap.

    Returns:
        The safely unwrapped item

    ToDo: Will be depecated. This function is pythonic but difficult to maintain the property of the
        xarray attrs.
    """
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
        input_dict (dict[str, Any]): input dictionary

    Returns:
        The unwrapped attributes as a dict.

    ToDo: Will be depecated. This function is pythonic but difficult to maintain the property of the
        xarray attrs.
    """
    return {k: unwrap_xarray_item(v) for k, v in input_dict.items()}


def apply_dataarray(
    arr: xr.DataArray,  # arr.values is used
    f: Callable[[NDArray[np.float_], Any], NDArray[np.float_]],
    *args: Incomplete,
    **kwargs: Incomplete,
) -> xr.DataArray:
    """Applies a function onto the values of a DataArray."""
    return xr.DataArray(
        f(arr.values, *args, **kwargs),
        arr.coords,
        arr.dims,
        attrs=arr.attrs,
    )


def lift_dataarray(  # unused
    f: Callable[[NDArray[np.float_], Any], NDArray[np.float_]],
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


def lift_dataarray_attrs(
    f: Callable[[dict[str, T], Any], dict[str, T]] | Callable[[dict[str, T]], dict[str, T]],
) -> Callable[[xr.DataArray], xr.DataArray]:
    """Lifts a function that operates dicts to a function that acts on dataarray attrs.

    Produces a new xr.DataArray.

    Args:
        f: Function to apply

    Returns:
        g: Function operating on the attributes of an xr.DataArray
    """

    def g(
        arr: xr.DataArray,
        *args: Incomplete,
        **kwargs: Incomplete,
    ) -> xr.DataArray:
        """[TODO:summary].

        Args:
            arr (xr.DataArray): ARPES Data
            *args: Pass to function f
            **kwargs: Pass to function f

        Returns:
            xr.DataArray
        """
        return xr.DataArray(
            arr.values,
            arr.coords,
            arr.dims,
            attrs=f(arr.attrs, *args, **kwargs),
        )

    return g


def lift_datavar_attrs(
    f: Callable[[dict[str, T], Any], dict[str, T]] | Callable[[dict[str, T]], dict[str, T]],
) -> Callable[..., DataType]:
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
        *args: Incomplete,
        **kwargs: Incomplete,
    ) -> DataType:
        """[TODO:summary].

        Args:
            data (DataType): ARPES Data
            *args: pass to arr_lifted & and function "f"
            **kwargs: pass to arr_lifted & and function "f"
        """
        arr_lifted = lift_dataarray_attrs(f)
        if isinstance(data, xr.DataArray):
            return arr_lifted(data, *args, **kwargs)

        var_names = list(data.data_vars.keys())
        new_vars = {k: arr_lifted(data[k], *args, **kwargs) for k in var_names}
        new_root_attrs = f(data.attrs, *args, **kwargs)

        return xr.Dataset(new_vars, data.coords, new_root_attrs)

    return g
