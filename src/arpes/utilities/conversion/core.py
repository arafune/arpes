"""Helper functions for coordinate transformations and user/analysis API.

All the functions here assume standard polar angles, as given in the
`data model documentation <https://arpes.readthedocs.io/spectra>`_.

Functions here must accept constants or numpy arrays as valid inputs,
so all standard math functions have been replaced by their equivalents out
of numpy. Array broadcasting should handle any issues or weirdnesses that
would encourage the use of direct iteration, but in case you need to write
a conversion directly, be aware that any functions here must work on arrays
as well for consistency with client code.

Everywhere:

Kinetic energy -> 'kinetic_energy'
Binding energy -> 'eV', for convenience (negative below 0)
Photon energy -> 'hv'

Better facilities should be added for ToFs to do simultaneous (timing, angle)
to (binding energy, k-space).
"""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Literal, TypeGuard

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from arpes.debug import setup_logger

from .fast_interp import Interpolator

if TYPE_CHECKING:
    from numpy.typing import NDArray


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


def grid_interpolator_from_dataarray(
    arr: xr.DataArray,
    fill_value: float = 0.0,
    method: Literal["linear", "nearest", "slinear", "cubic", "quintic", "pchip"] = "linear",
    *,
    bounds_error: bool = False,
) -> RegularGridInterpolator | Interpolator:
    """Translates an xarray.DataArray contents into a scipy.interpolate.RegularGridInterpolator.

    This is principally used for coordinate translations.
    """
    assert isinstance(arr, xr.DataArray)
    flip_axes: set[str] = set()
    for d in arr.dims:
        c = arr.coords[d]
        if len(c) > 1 and c[1] - c[0] < 0:
            flip_axes.add(str(d))
    values: NDArray[np.floating] = arr.values
    for dim in flip_axes:
        values = np.flip(values, arr.dims.index(dim))
    interp_points: list[NDArray[np.floating]] = [
        arr.coords[d].values[::-1] if d in flip_axes else arr.coords[d].values for d in arr.dims
    ]
    trace_size = [len(pts) for pts in interp_points]

    if method == "linear":
        logger.debug(f"Using fast_interp.Interpolator: size {trace_size}")
        return Interpolator.from_arrays(interp_points, values)
    return RegularGridInterpolator(
        points=tuple(interp_points),
        values=values,
        bounds_error=bounds_error,
        fill_value=fill_value,
        method=method,
    )


def _is_dims_match_coordinate_convert(
    angles: tuple[str, ...],
) -> TypeGuard[
    tuple[Literal["phi"]]
    | tuple[Literal["beta"], Literal["phi"]]
    | tuple[Literal["phi"], Literal["theta"]]
    | tuple[Literal["phi"], Literal["psi"]]
    | tuple[Literal["hv"], Literal["phi"]]
]:
    return angles in {
        ("phi",),
        ("beta", "phi"),
        ("phi", "theta"),
        ("phi", "psi"),
        ("hv", "phi"),
    }
