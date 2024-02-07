"""Very basic, generic time-resolved ARPES analysis tools."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from arpes.preparation import normalize_dim
from arpes.provenance import update_provenance
from arpes.utilities import normalize_to_spectrum

if TYPE_CHECKING:
    from arpes._typing import DataType

__all__ = ("find_t0", "relative_change", "normalized_relative_change")


@update_provenance("Normalized subtraction map")
def normalized_relative_change(
    data: xr.DataArray | xr.Dataset,
    t0: float | None = None,
    buffer: float = 0.3,
    *,
    normalize_delay: bool = True,
) -> xr.DataArray:
    """Calculates a normalized relative Tr-ARPES change in a delay scan.

    Obtained by normalizing along the pump-probe "delay" axis and then subtracting
    the mean before t0 data and dividing by the original spectrum.

    Args:
        data: The input spectrum to be normalized. Should have a "delay" dimension.
        t0: The t0 for the input array.
        buffer: How far before t0 to select equilibrium data. Should be at least
          the temporal resolution in ps.
        normalize_delay: If true, normalizes data along the "delay" dimension.

    Returns:
        The normalized data.
    """
    spectrum = normalize_to_spectrum(data)
    assert isinstance(spectrum, xr.DataArray)
    if normalize_delay:
        spectrum = normalize_dim(spectrum, "delay")
    subtracted = relative_change(spectrum, t0, buffer, normalize_delay=False)
    assert isinstance(subtracted, xr.DataArray)
    normalized: xr.DataArray = subtracted / spectrum
    normalized.values[np.isinf(normalized.values)] = 0
    normalized.values[np.isnan(normalized.values)] = 0
    return normalized


@update_provenance("Created simple subtraction map")
def relative_change(
    data: xr.Dataset | xr.DataArray,
    t0: float | None = None,
    buffer: float = 0.3,
    *,
    normalize_delay: bool = True,
) -> xr.DataArray:
    """Like normalized_relative_change, but only subtracts the before t0 data.

    Args:
        data: The input spectrum to be normalized. Should have a "delay" dimension.
        t0: The t0 for the input array.
        buffer: How far before t0 to select equilibrium data. Should be at least
          the temporal resolution in ps.
        normalize_delay: If true, normalizes data along the "delay" dimension.

    Returns:
        The normalized data.
    """
    spectrum = normalize_to_spectrum(data)
    assert isinstance(spectrum, xr.DataArray)
    if normalize_delay:
        spectrum = normalize_dim(spectrum, "delay")

    delay_coords = spectrum.coords["delay"]
    delay_start = np.min(delay_coords)

    if t0 is None:
        t0 = find_t0(spectrum)
    assert t0 is not None
    assert t0 - buffer > delay_start

    before_t0 = spectrum.sel(delay=slice(None, t0 - buffer))
    return spectrum - before_t0.mean("delay")


def find_t0(data: DataType, e_bound: float = 0.02) -> float:
    """Finds the effective t0 by fitting excited carriers.

    Args:
        data: A spectrum with "eV" and "delay" dimensions.
        e_bound: Lower bound on the energy to use for the fitting

    Returns:
        The delay value at the estimated t0.

    """
    warnings.warn(
        "This function will be deprecated, because it's not so physically correct.",
        stacklevel=2,
    )
    spectrum = normalize_to_spectrum(data)
    assert isinstance(spectrum, xr.DataArray)
    assert "delay" in spectrum.dims
    assert "eV" in spectrum.dims

    if "t0" in spectrum.attrs:
        return float(spectrum.attrs["t0"])
    if "T0_ps" in spectrum.attrs:
        return float(spectrum.attrs["T0_ps"])
    sum_dims = set(spectrum.dims)
    sum_dims.remove("delay")
    sum_dims.remove("eV")

    summed = spectrum.sum(list(sum_dims)).sel(eV=slice(e_bound, None)).mean("eV")
    coord_max = summed.argmax().item()
    return summed.coords["delay"].values[coord_max]
