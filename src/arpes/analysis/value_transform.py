"""Utilities related to function application on values of dataarray."""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Any

from arpes.debug import setup_logger

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    import xarray as xr
    from _typeshed import Incomplete
    from numpy.typing import NDArray


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
