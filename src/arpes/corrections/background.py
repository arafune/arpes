"""For estimating the above Fermi level incoherent background."""

from __future__ import annotations

import numpy as np
import xarray as xr

from arpes.provenance import update_provenance
from arpes.utilities import normalize_to_spectrum

__all__ = ("remove_incoherent_background",)


@update_provenance("Remove incoherent background from above Fermi level")
def remove_incoherent_background(
    data: xr.DataArray,
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
        set_zero (bool): set zero if the negative value is obtained after background subtraction.

    Returns:
        Data with a background subtracted.
    """
    data_array = data if isinstance(data, xr.DataArray) else normalize_to_spectrum(data)

    approximate_fermi_energy_level = data_array.S.find_spectrum_energy_edges().max()

    background = data_array.sel(eV=slice(approximate_fermi_energy_level + 0.1, None))
    density = background.sum("eV") / (np.logical_not(np.isnan(background)) * 1).sum("eV")
    new = data_array - density
    if set_zero:
        new.values[new.values < 0] = 0

    return new
