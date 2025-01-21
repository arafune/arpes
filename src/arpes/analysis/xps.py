"""X-ray photoelectron spectroscopy related analysis.

Primarily, curve fitting and peak-finding utilities for XPS.
"""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from arpes.debug import setup_logger
from arpes.utilities import normalize_to_spectrum

from .filters import savitzky_golay_filter
from .general import rebin

if TYPE_CHECKING:
    from numpy.typing import NDArray


__all__ = ("approximate_core_levels",)

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


def local_minima(a: NDArray[np.float64], promenance: int = 3) -> NDArray[np.bool_]:
    """Calculates local minima (maxima) according to a prominence criterion.

    The point should be lower than any in the region around it.

    Rather than searching manually, we perform some fancy indexing to do the calculation
    across the whole array simultaneously and iterating over the promenance criterion instead of
    through the data and the promenance criterion.

    Args:
        a: The input array to calculate local minima over
        promenance: The prominence over indices required to be called a local minimum

    Returns:
        A mask where the local minima are True and other values are False.
    """
    conditions = ~np.zeros_like(a, dtype=bool)
    for i in range(1, promenance + 1):
        current_conditions = np.r_[[False] * i, a[i:] < a[:-i]] & np.r_[a[:-i] < a[i:], [False] * i]
        conditions = conditions & current_conditions

    return conditions


def local_maxima(a: NDArray[np.float64], promenance: int = 3) -> NDArray[np.bool_]:
    return local_minima(-a, promenance)


local_maxima.__doc__ = local_minima.__doc__


def approximate_core_levels(
    data: xr.DataArray,
    window_length: int = 0,
    polyorder: int = 2,
    binning: int = 3,
    promenance: int = 5,
) -> list[float]:
    """Approximately locates core levels in a spectrum.

    Data is first smoothed, and then local maxima with sufficient prominence over
    other nearby points are selected as peaks.

    This can be helfpul to "seed" a curve fitting analysis for XPS.

    Args:
        data: An XPS spectrum.
        window_length: Savitzky-Golay window size (should be >= 5)
        polyorder: Savitzky-Golay order (2 is usual and sufficient in most case)
        binning: Used for approximate smoothing
        promenance: Required promenance over nearby peaks

    Returns:
        A set of energies with candidate peaks.
    """
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)

    dos = data.S.sum_other(["eV"])

    if not window_length:
        window_length = int(len(dos) / 40)  # empirical, may change
        if window_length % 2 == 0:
            window_length += 1
    logger.debug(f"window_length: {window_length}")
    smoothed = rebin(
        savitzky_golay_filter(
            data=dos,
            window_length=window_length,
            polyorder=polyorder,
        ),
        eV=binning,
    )

    indices = np.argwhere(local_maxima(smoothed.values, promenance=promenance))
    return [smoothed.coords["eV"][idx].item() for idx in indices]
