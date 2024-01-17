"""Implements data loading for the Lanzara group "Main Chamber"."""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np

import arpes.xarray_extensions  # pylint: disable=unused-import, redefined-outer-name  # noqa: F401
from arpes.endstations import SCANDESC, FITSEndstation, HemisphericalEndstation

if TYPE_CHECKING:
    import xarray as xr

    from arpes._typing import ARPESAttrs

__all__ = ("ALGMainChamber",)


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

    ATTR_TRANSFORMS: ClassVar[dict] = {
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

    MERGE_ATTRS: ClassVar[ARPESAttrs] = {
        "analyzer": "Specs PHOIBOS 150",
        "analyzer_name": "Specs PHOIBOS 150",
        "parallel_deflectors": False,
        "perpendicular_deflectors": False,
        "analyzer_radius": 150,
        "analyzer_type": "hemispherical",
        "mcp_voltage": None,
        "probe_linewidth": 0.015,
    }

    def postprocess_final(
        self,
        data: xr.Dataset,
        scan_desc: SCANDESC | None = None,
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
        for spectrum in data.S.spectra:
            spectrum.attrs["hv"] = 5.93  # only photon energy available on this chamber
            spectrum.attrs["alpha"] = 0
            spectrum.attrs["psi"] = 0
            spectrum.attrs["phi_offset"] = 0.405

        data = super().postprocess_final(data, scan_desc)

        if "beta" in data.coords:
            data = data.assign_coords(beta=np.deg2rad(data.beta.values))

        return data
