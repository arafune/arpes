"""For estimating the above Fermi level incoherent background."""

from __future__ import annotations

from logging import DEBUG, INFO

import numpy as np
import xarray as xr

from arpes.debug import setup_logger
from arpes.provenance import update_provenance
from arpes.utilities import normalize_to_spectrum

__all__ = ("remove_incoherent_background",)

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


@update_provenance("Remove incoherent background from above Fermi level")
def remove_incoherent_background(
    data: xr.DataArray,
    fermi_level: float | None = None,
    *,
    set_zero: bool = True,
) -> xr.DataArray:
    """Removes counts above the Fermi level.

    Sometimes spectra are contaminated by data above the Fermi level for
    various reasons (such as broad core levels from 2nd harmonic light,
    or slow enough electrons in ToF experiments to be counted in subsequent
    pulses).

    Args:
        data (XrTypes): input ARPES data
        fermi_level (float | None): Fermi level, if not set, estimate it internally.
        set_zero (bool): set zero if the negative value is obtained after background subtraction.

    Returns:
        Data with a background subtracted.
    """
    data = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)
    if fermi_level is None:
        fermi_level = data.S.find_spectrum_energy_edges().max()
    assert isinstance(fermi_level, float)
    logger.debug(f"fermi_level: {fermi_level}")

    background = data.sel(eV=slice(fermi_level + 0.1, None))
    density = background.sum("eV") / (np.logical_not(np.isnan(background)) * 1).sum("eV")
    new = data - density
    if set_zero:
        new.values[new.values < 0] = 0

    return new
