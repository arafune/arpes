"""Very basic, generic time-resolved ARPES analysis tools."""

from __future__ import annotations

from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from arpes.debug import setup_logger
from arpes.preparation import normalize_dim
from arpes.preparation.axis_preparation import vstack_data
from arpes.provenance import update_provenance
from arpes.utilities import normalize_to_spectrum

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = (
    "build_crosscorrelation",
    "delaytime_fs",
    "find_t_for_max_intensity",
    "position_mm_to_delaytime_fs",
    "relative_change",
)


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


def delaytime_fs(
    mirror_movement: float,
    units: Literal["um", "mm", "nm", "μm", "AA", "Å"] = "um",
) -> float:
    """Return delaytime from the mirror movement (not position).

    Args:
        mirror_movement (float): mirror movement in micron unit.
        units: Units for mirror movement. default to um (μm).

    >>> delaytime_fs(10)
    33.35640951981521

    Returns: float
        delay time in fs.
    """
    return (
        3.335640951981521
        * mirror_movement
        * {"um": 1, "mm": 1e3, "nm": 1e-3, "AA": 1e-4, "μm": 1, "Å": 1e-4}[units]
    )


def position_mm_to_delaytime_fs(
    position_mm: float,
) -> float:
    """Return delay time from the mirror position.

    Args:
        position_mm (np.ndarray | float): mirror position

    Returns: np.ndarray | float
        delay time in fs unit.

    """
    return delaytime_fs(2 * position_mm, "mm")


def build_crosscorrelation(
    datalist: Sequence[xr.DataArray],
    delayline_dim: str = "position",
    delayline_origin: float = 0,
    *,
    convert_position_to_time: Callable[[float], float] | None = position_mm_to_delaytime_fs,
) -> xr.DataArray:
    """Constructs a multidimensional data array from cross-correlation measurements.

    This function processes a series of cross-correlation data arrays by assigning delay
    times based on the specified delay line dimension. It supports conversion from
    position units (e.g., mm) to time units if requested.

    Args:
        datalist (Sequence[xr.DataArray]):
            Data series from the cross-correlation experiments. Each data element should contain the
            delay line value in attrs[delayline_dim], not in coolrds.
        delayline_dim(str, optional):
            The key in data.attrs representing the delay line value (default: "position").
            When this is "position", the unit is assumed to be mm.
        delayline_origin (float, optional):
            The value corresponding to the delay zero position.  Defaults to 0.
        convert_position_to_time (Callable[[float], float] | None):
            Function to convert the delay line values from position to time units. Default to
            position_mm_to_delaytime_fs, which convert to the delayline position in mm to delay time
            in fs.  For example, when you need convert to delay time in ps,  another conversion
            function is required. If None, the delay line values are used as-is.

    Returns: xr.DataArray
        A stacked data array with an additional "delay" dimension.
    """
    cross_correlations = []

    for spectrum in datalist:
        spectrum_arr = (
            spectrum if isinstance(spectrum, xr.DataArray) else normalize_to_spectrum(spectrum)
        )

        raw_value = float(spectrum_arr.attrs[delayline_dim])
        if convert_position_to_time:
            delay_time = convert_position_to_time(raw_value) - convert_position_to_time(
                delayline_origin,
            )
        else:
            delay_time = raw_value - delayline_origin

        cross_correlations.append(
            spectrum_arr.assign_coords({"delay": delay_time}).expand_dims("delay"),
        )

    return vstack_data(
        cross_correlations,
        new_dim="delay",
        sort=True,
    )


@update_provenance("Relative change map")
def relative_change(
    data: xr.DataArray,
    t0: float | None = None,
    buffer_fs: float = 300,
    *,
    normalize_delay: bool = True,
    divide_by_spectrum: bool = False,
) -> xr.DataArray:
    """Calculate relative Tr-ARPES change in a delay scan.

    Subtracts the mean of the pre-t0 spectrum. Optionally, the result is
    divided by the original spectrum.

    Args:
        data: Input spectrum. Should have a "delay" dimension.
        t0: Time-zero (fs). If None, determined automatically.
        buffer_fs: Width (fs) of pre-t0 region used as equilibrium reference.
        normalize_delay: If true, normalize along "delay" dimension before processing.
        divide_by_spectrum: If true, divide subtracted spectrum by the original spectrum
            (like normalized_relative_change).

    Returns:
        xr.DataArray: Relative (and optionally normalized) change map.
    """
    # Ensure DataArray + optional delay normalization
    spectrum = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    assert isinstance(spectrum, xr.DataArray)
    if normalize_delay:
        spectrum = normalize_dim(spectrum, "delay")

    # Determine t0 and pre-t0 region
    delay_start: float = np.min(spectrum.coords["delay"]).values.item()
    if t0 is None:
        t0 = find_t_for_max_intensity(spectrum)
    assert t0 is not None
    assert t0 - buffer_fs > delay_start

    # Subtraction
    before_t0 = spectrum.sel(delay=slice(None, t0 - buffer_fs))
    subtracted = spectrum - before_t0.mean("delay", keep_attrs=True)

    # Optional division
    if divide_by_spectrum:
        subtracted = subtracted / spectrum
        subtracted = xr.where(np.isfinite(subtracted), subtracted, 0)
    subtracted.attrs["subtracted"] = True
    return subtracted


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

    e_slice = slice(*e_bounds) if any(e_bounds) else slice(None)
    summed = data.sum(list(sum_dims)).sel(eV=e_slice).mean("eV")
    return summed.idxmax().item()
