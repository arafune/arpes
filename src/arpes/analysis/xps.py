"""X-ray photoelectron spectroscopy related analysis.

Primarily, curve fitting and peak-finding utilities for XPS.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from arpes.utilities import normalize_to_spectrum

from .general import rebin
from .savitzky_golay import savitzky_golay

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from arpes._typing import DataType

__all__ = ("approximate_core_levels",)


def local_minima(a: NDArray[np.float_], promenance: int = 3) -> NDArray[np.float_]:
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
    conditions = a == a
    for i in range(1, promenance + 1):
        current_conditions = np.r_[[False] * i, a[i:] < a[:-i]] & np.r_[a[:-i] < a[i:], [False] * i]
        conditions = conditions & current_conditions

    return conditions


def local_maxima(a: NDArray[np.float_], promenance: int = 3) -> NDArray[np.float_]:
    return local_minima(-a, promenance)


local_maxima.__doc__ = local_minima.__doc__


def approximate_core_levels(
    data: DataType,
    window_size: int = 0,
    order: int = 5,
    binning: int = 3,
    promenance: int = 5,
) -> list[float]:
    """Approximately locates core levels in a spectrum.

    Data is first smoothed, and then local maxima with sufficient prominence over
    other nearby points are selected as peaks.

    This can be helfpul to "seed" a curve fitting analysis for XPS.

    Args:
        data: An XPS spectrum.
        window_size: Savitzky-Golay window size
        order: Savitzky-Golay order
        binning: Used for approximate smoothing
        promenance: Required promenance over nearby peaks

    Returns:
        A set of energies with candidate peaks.
    """
    data_array = normalize_to_spectrum(data)

    dos = data_array.S.sum_other(["eV"]).sel(eV=slice(None, -20))

    if not window_size:
        window_size = int(len(dos) / 40)  # empirical, may change
        if window_size % 2 == 0:
            window_size += 1
    smoothed = rebin(savitzky_golay(dos, window_size, order), eV=binning)

    indices = np.argwhere(local_maxima(smoothed.values, promenance=promenance))
    return [smoothed.coords["eV"][idx].item() for idx in indices]
