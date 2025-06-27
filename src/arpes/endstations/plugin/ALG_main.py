"""Implements data loading for the Lanzara group "Main Chamber"."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pint

from arpes.endstations import FITSEndstation, HemisphericalEndstation
from arpes.laser import electrons_per_pulse

if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr

    from arpes._typing import ScanDesc, Spectrometer

__all__ = ("ALGMainChamber", "electrons_per_pulse_mira")


class ALGMainChamber(HemisphericalEndstation, FITSEndstation):
    """Implements data loading for the Lanzara group "Main Chamber"."""

    PRINCIPAL_NAME = "ALG-Main"
    ALIASES: ClassVar[list[str]] = [
        "MC",
        "ALG-Main",
        "ALG-MC",
        "ALG-Hemisphere",
        "ALG-Main Chamber",
    ]

    ATTR_TRANSFORMS: ClassVar[dict[str, Callable[..., dict[str, float | list[str] | str]]]] = {
        "START_T": lambda _: {"time": " ".join(_.split(" ")[1:]).lower(), "date": _.split(" ")[0]},
    }

    RENAME_KEYS: ClassVar[dict[str, str]] = {
        "Phi": "chi",
        "Beta": "beta",
        "Theta": "theta",
        "Azimuth": "chi",
        "Alpha": "alpha",
        "Pump_energy_uJcm2": "pump_fluence",
        "T0_ps": "t0_nominal",
        "W_func": "workfunction",
        "Slit": "slit",
        "LMOTOR0": "x",
        "LMOTOR1": "y",
        "LMOTOR2": "z",
        "LMOTOR3": "theta",
        "LMOTOR4": "beta",
        "LMOTOR5": "chi",
        "LMOTOR6": "delay",
        "SFLNM0": "lens_mode_name",
        "SFFR_0": "frames_per_slice",
        "SFBA_0": "phi_prebinning",
        "SFBE0": "eV_prebinning",
    }

    MERGE_ATTRS: ClassVar[Spectrometer] = {
        "analyzer": "Specs PHOIBOS 150",
        "analyzer_name": "Specs PHOIBOS 150",
        "parallel_deflectors": False,
        "perpendicular_deflectors": False,
        "analyzer_radius": 150,
        "analyzer_type": "hemispherical",
        "mcp_voltage": np.nan,
        "probe_linewidth": 0.015,
    }

    def postprocess_final(
        self,
        data: xr.Dataset,
        scan_desc: ScanDesc | None = None,
    ) -> xr.Dataset:
        """Performs final normalization of scan data.

        For the Lanzaa group main chamber, this means:

        1. Associating the fixex UV laser energy.
        2. Adding missing coordinates.
        3. Using a standard approximate set of coordinate offsets.
        4. Converting relevant angular coordinates to radians.
        """
        data.attrs["hv"] = 5.93
        data.attrs["alpha"] = 0
        data.attrs["psi"] = 0

        # by default we use this value since this isnear the center of the spectrometer window
        data.attrs["phi_offset"] = 0.405
        for spectrum in data.data_vars.values():
            spectrum.attrs["hv"] = 5.93  # only photon energy available on this chamber
            spectrum.attrs["alpha"] = 0
            spectrum.attrs["psi"] = 0
            spectrum.attrs["phi_offset"] = 0.405

        data = super().postprocess_final(data, scan_desc)

        if "beta" in data.coords:
            data = data.assign_coords(beta=np.deg2rad(data.beta.values))

        return data


ureg = pint.UnitRegistry()


def electrons_per_pulse_mira(photocurrent: pint.Quantity, division_ratio: int = 1) -> float:
    """Specific case of `electrons_per_pulse` for Mira oscillators.

    Originally, this function was in laser.py. However, it moved here because it  is useful only for
    the group of the original author (At least the repetition ration of our Mira is different from
    the value below).

    Args:
        photocurrent: [TODO:description]
        division_ratio: [TODO:description]

    Returns: (float)
        [TODO:description]
    """
    mira_frequency = 54.3 / ureg.microsecond
    return electrons_per_pulse(photocurrent, mira_frequency, division_ratio)
