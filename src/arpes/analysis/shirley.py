"""Contains routines for calculating and removing the classic Shirley background."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, TypedDict, Unpack

import numpy as np
import xarray as xr

from arpes.provenance import update_provenance
from arpes.utilities import normalize_to_spectrum

if TYPE_CHECKING:
    from numpy.typing import NDArray


__all__ = (
    "calculate_shirley_background",
    "calculate_shirley_background_full_range",
    "remove_shirley_background",
)


class KwargsCalShirleyBGFunc(TypedDict, total=False):
    energy_range: slice | None
    eps: float
    max_iters: int
    n_samples: int


@update_provenance("Remove Shirley background")
def remove_shirley_background(
    xps: xr.DataArray,
    **kwargs: Unpack[KwargsCalShirleyBGFunc],
) -> xr.DataArray:
    """Calculates and removes a Shirley background from a spectrum.

    Only the background corrected spectrum is retrieved.

    Args:
        xps: The input array.
        kwargs: Parameters to feed to the background estimation routine.

    Returns:
        The the input array with a Shirley background subtracted.
    """
    xps_array = xps if isinstance(xps, xr.DataArray) else normalize_to_spectrum(xps)
    return xps_array - calculate_shirley_background(xps_array, **kwargs)


def _calculate_shirley_background_full_range(
    xps: NDArray[np.float64],
    eps: float = 1e-7,
    max_iters: int = 50,
    n_samples: int = 5,
) -> NDArray[np.float64]:
    """Core routine for calculating a Shirley background on np.ndarray data."""
    background = np.copy(xps)
    cumulative_xps = np.cumsum(xps, axis=0)
    total_xps = np.sum(xps, axis=0)
    rel_error = np.inf

    i_left = np.mean(xps[:n_samples], axis=0)
    i_right = np.mean(xps[-n_samples:], axis=0)

    iter_count = 0

    k = i_left - i_right
    for _ in range(max_iters):
        cumulative_background = np.cumsum(background, axis=0)
        total_background = np.sum(background, axis=0)

        new_bkg = np.copy(background)

        for i in range(len(new_bkg)):
            new_bkg[i] = i_right + k * (
                (total_xps - cumulative_xps[i] - (total_background - cumulative_background[i]))
                / (total_xps - total_background + 1e-5)
            )

        rel_error = np.abs(np.sum(new_bkg, axis=0) - total_background) / (total_background)

        background = new_bkg

        if np.any(rel_error < eps):
            break

    if (iter_count + 1) == max_iters:
        msg = (
            "Shirley background calculation did not converge "
            f"after {max_iters} steps with relative error {rel_error}!"
        )
        warnings.warn(msg, stacklevel=2)

    return background


@update_provenance("Calculate full range Shirley background")
def calculate_shirley_background_full_range(
    xps: xr.DataArray,
    eps: float = 1e-7,
    max_iters: int = 50,
    n_samples: int = 5,
) -> xr.DataArray:
    """Calculates a shirley background.

    The background is defined according to:

    S(E) = I(E_right) + k * (A_right(E)) / (A_left(E) + A_right(E))

    Typically

    k := I(E_right) - I(E_left)

    The iterative method is continued so long as the total background is not converged to relative
    error `eps`.

    The method continues for a maximum number of iterations `max_iters`.

    In practice, what we can do is to calculate the cumulative sum of the data along the energy axis
    of both the data and the current estimate of the background

    Args:
        xps: The input data.
        eps: Convergence parameter.
        max_iters: The maximum number of iterations to allow before convengence.
        n_samples: The number of samples to use at the boundaries of the input data.

    Returns:
        A monotonic Shirley background over the entire energy range.
    """
    xps_array = (
        xps.copy(deep=True)
        if isinstance(xps, xr.DataArray)
        else normalize_to_spectrum(xps).copy(deep=True)
    )
    core_dims = [d for d in xps_array.dims if d != "eV"]

    return xr.apply_ufunc(
        _calculate_shirley_background_full_range,
        xps_array,
        eps,
        max_iters,
        n_samples,
        input_core_dims=[core_dims, [], [], []],
        output_core_dims=[core_dims],
        exclude_dims=set(core_dims),
        vectorize=False,
    )


@update_provenance("Calculate limited range Shirley background")
def calculate_shirley_background(
    xps: xr.DataArray,
    energy_range: slice | None = None,
    eps: float = 1e-7,
    max_iters: int = 50,
    n_samples: int = 5,
) -> xr.DataArray:
    """Calculates a shirley background iteratively over the full energy range `energy_range`.

    Uses `calculate_shirley_background_full_range` internally.

    Outside the indicated range, the background is extrapolated as a constant from
    the nearest in-range value.

    Args:
        xps: The input data.
        energy_range: A slice with the energy range to be used.
        eps: Convergence parameter.
        max_iters: The maximum number of iterations to allow before convengence.
        n_samples: The number of samples to use at the boundaries of the input data.

    Returns:
        A monotonic Shirley background over the entire energy range.
    """
    if energy_range is None:
        energy_range = slice(None, None)

    xps_array = xps if isinstance(xps, xr.DataArray) else normalize_to_spectrum(xps)
    assert isinstance(xps_array, xr.DataArray)
    xps_for_calc = xps_array.sel(eV=energy_range)

    bkg = calculate_shirley_background_full_range(xps_for_calc, eps, max_iters, n_samples)
    bkg = bkg.transpose(*xps_array.dims)
    full_bkg = xps_array * 0

    left_idx = np.searchsorted(full_bkg.eV.values, bkg.eV.values[0], side="left")
    right_idx = left_idx + len(bkg)

    full_bkg.values[:left_idx] = bkg.values[0]
    full_bkg.values[left_idx:right_idx] = bkg.values
    full_bkg.values[right_idx:] = bkg.values[-1]

    return full_bkg
