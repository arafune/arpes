"""Very basic, generic time-resolved ARPES analysis tools."""

from __future__ import annotations

from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

from arpes.preparation import normalize_dim
from arpes.provenance import update_provenance
from arpes.utilities import normalize_to_spectrum

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = (
    "build_crosscorrelation",
    "delaytime_fs",
    "find_t_for_max_intensity",
    "normalized_relative_change",
    "position_to_delaytime",
    "relative_change",
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


def build_crosscorrelation(
    datalist: Sequence[xr.DataArray],
    delayline_dim: str = "position",
    delayline_origin: float = 0,
    *,
    convert_position_to_time: bool = True,
) -> xr.DataArray:
    """Constructs a multidimensional data array from cross-correlation measurements.

    This function processes a series of cross-correlation data arrays by assigning delay
    times based on the specified delay line dimension. It supports conversion from
    position units (e.g., mm) to time units if requested.

    Args:
        datalist (Sequence[xr.DataArray]):
            Data series from the cross-correlation experiments.
        delayline_dim(str, optional):
            The dimension name for "delay line", which must be in key of data.attrs
            When this is the "position" dimension, the unit is assumed to be "mm". If the value has
            already been converted to "time" dimension, set convert_position_to_time=True
        delayline_origin (float, optional):
            The value corresponding to the delay zero position.  Defaults to 0.
        convert_position_to_time (bool):
            Whether to convert the delay line values from position to time units. If set
            to False, the delay line values are used as-is. Defaults to True.

    Returns: xr.DataArray
    """
    cross_correlations = []

    for spectrum in datalist:
        spectrum_arr = (
            spectrum if isinstance(spectrum, xr.DataArray) else normalize_to_spectrum(spectrum)
        )
        if convert_position_to_time:
            delay_time = position_to_delaytime(
                float(spectrum_arr.attrs[delayline_dim]),
                delayline_origin,
            )
        else:
            delay_time = spectrum_arr.attrs[delayline_dim] - delayline_origin
        cross_correlations.append(
            spectrum_arr.assign_coords({"delay": delay_time}).expand_dims("delay"),
        )
    cross_correlation: xr.DataArray = xr.concat(cross_correlations, dim="delay")
    del cross_correlation.attrs[delayline_dim]
    return cross_correlation


def delaytime_fs(mirror_movement_um: float) -> float:
    """Return delaytime from the mirror movement (not position).

    Args:
        mirror_movement_um (float): mirror movement in micron unit.

    >>> delaytime_fs(10)
    33.35640951981521

    Returns: float
        delay time in fs.

    """
    return 3.335640951981521 * mirror_movement_um


def position_to_delaytime(position_mm: float, delayline_offset_mm: float) -> float:
    """Return delay time from the mirror position.

    Args:
        position_mm (np.ndarray | float): mirror position
        delayline_offset_mm (float): mirror position corresponding to the zero delay

    Returns: np.ndarray | float
        delay time in fs unit.

    """
    return delaytime_fs(2 * (position_mm - delayline_offset_mm) * 1000)


@update_provenance("Normalized subtraction map")
def normalized_relative_change(
    data: xr.DataArray,
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
    spectrum = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
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
    data: xr.DataArray,
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
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    assert isinstance(data, xr.DataArray)
    if normalize_delay:
        data = normalize_dim(data, "delay")

    delay_coords = data.coords["delay"]
    delay_start = np.min(delay_coords)

    if t0 is None:
        t0 = find_t_for_max_intensity(data)
    assert t0 is not None
    assert t0 - buffer > delay_start

    before_t0 = data.sel(delay=slice(None, t0 - buffer))
    return data - before_t0.mean("delay")


def find_t_for_max_intensity(
    data: xr.DataArray,
    e_bounds: tuple[float | None, float | None] = (None, None),
) -> float:
    """Finds the time corresponding to the maximum (integrated) intensity.

    While the time returned can be used to "t=0" in pump probe exepriments, especially for
    relatively slow (~ps) phenomena, but not always true.

    Args:
        data: A spectrum with "eV" and "delay" dimensions.
        e_bounds: Lower and Higher bound on the energy to use for the fitting

    Returns:
        The  value at the estimated t0.

    """
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    assert isinstance(data, xr.DataArray)
    assert "delay" in data.dims
    assert "eV" in data.dims
    sum_dims = set(data.dims)
    sum_dims.remove("delay")
    sum_dims.remove("eV")

    summed = data.sum(list(sum_dims)).sel(eV=slice(e_bounds[0], e_bounds[1])).mean("eV")
    return summed.idxmax().item()
