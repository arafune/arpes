"""Implements data loading for the Phelix beamline @ Solaris.

The plugin supports flowing scenarios:
- Loading single maps.
- Loading 3D map measured with deflector.
- Loading 3D map measured using manipulator rotation by polar angle.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr

from arpes.endstations import (
    HemisphericalEndstation,
    SingleFileEndstation,
    SynchrotronEndstation,
    add_endstation,
)
from arpes.endstations.prodigy_xy import load_xy

if TYPE_CHECKING:
    from collections.abc import Callable

    from arpes._typing import Spectrometer
    from arpes.endstations import ScanDesc

__all__ = ["Phelix"]


class Phelix(HemisphericalEndstation, SingleFileEndstation, SynchrotronEndstation):
    """Implements loading xy text files from the Specs Prodigy software."""

    PRINCIPAL_NAME = "Phelix"
    ALIASES: ClassVar[list[str]] = ["Phelix", "Phelix_JU", "Phelix_Solaris"]

    _TOLERATED_EXTENSIONS: ClassVar[set[str]] = {".xy"}

    # Mapping of lens modes to dispersion or magnification modes
    _LENS_MAPPING: ClassVar[dict[str, bool]] = {
        "HighAngularDispersion": True,
        "MediumAngularDispersion": True,
        "LowAngularDispersion": True,
        "MediumAngleMode": True,
        "WideAngleMode": True,
        "LowMagnification": False,
        "MediumMagnification": False,
        "HighMagnification": False,
    }

    # Angle values of the manipulator that correspond to normal emission
    _NORMAL_EMISSION: ClassVar[dict[str, float]] = {
        "anr1": 83.5,
    }

    RENAME_KEYS: ClassVar[dict[str, str]] = {
        "eff_workfunction": "workfunction",
        "acquisition_date": "aquisition_date_utc",
        "analyzer_slit": "slit",
        "analyzer_lens": "lens_mode",
        "detector_voltage": "mcp_voltage",
        "excitation_energy": "hv",
        "region": "id",
        "shiftx": "psi",
        "anr1": "theta",
    }

    ATTR_TRANSFORMS: ClassVar[dict[str, Callable[..., dict[str, float | list[str] | str]]]] = {
        "aquisition_date_utc": lambda _: {
            "date": _.split()[0],
            "time": _.split()[1],
        },
        "slit": lambda _: {
            "slit_number": int(_.split(":")[0]),
            "slit_width": float(_.split(":")[1].split("x")[0]),
        },
    }


    MERGE_ATTRS: ClassVar[Spectrometer] = {
        "analyzer": "Specs PHOIBOS 225",
        "analyzer_name": "Specs PHOIBOS 225",
        "parallel_deflectors": True,
        "perpendicular_deflectors": True,
        "analyzer_radius": 225,
        "analyzer_type": "hemispherical",
    }

    def load_single_frame(
        self,
        frame_path: str | Path = "",
        scan_desc: ScanDesc | None = None,
        **kwargs: str | float,
    ) -> xr.Dataset:
        """Load single xy file."""
        if scan_desc is None:
            scan_desc = {}
        file = Path(frame_path)
        if file.suffix in self._TOLERATED_EXTENSIONS:
            data = load_xy(frame_path, **kwargs)

            if "anr1" in data.coords:
                # Invert the anr1 manipulator axis and shift it to get theta angle
                data = data.assign_coords(
                    anr1 = -data.anr1 - Phelix._NORMAL_EMISSION["anr1"])
                data = data.isel(anr1=slice(None, None, -1))

            return xr.Dataset({"spectrum": data}, attrs=data.attrs)

        msg = "Data file must be ended with .xy"
        raise RuntimeError(msg)

    def postprocess_final(
        self,
        data: xr.Dataset,
        scan_desc: ScanDesc | None = None,
    ) -> xr.Dataset:
        """Perform final processing on the ARPES data.

        - Calculate phi or x values depending on the lens mode.
        - Add missing parameters.
        - Rename keys and dimensions in particular the third dimension that
        could be psi angle or theta angle in this endstation.

        Args:
            data(xr.Dataset): ARPES data
            scan_desc(SCANDESC | None): scan_description. Not used currently

        Returns:
            xr.Dataset: pyARPES compatible.
        """
        # Convert to binding energy notation
        binding_energies = data.coords["eV"].values - data.attrs["hv"]
        data = data.assign_coords({"eV": binding_energies})

        # Calculate phi or x values depending on the lens mode.
        lens_mode = data.attrs["lens_mode"].split(":")[0]
        if lens_mode in self._LENS_MAPPING:
            dispersion_mode = self._LENS_MAPPING[lens_mode]
            if dispersion_mode:
                data = data.rename({"nonenergy": "phi"})
            else:
                data = data.rename({"nonenergy": "x"})
        else:
            msg = f"Unknown Analyzer Lens: {lens_mode}"
            raise ValueError(msg)

        # Add missing parameters
        if scan_desc is None:
            scan_desc = {}
        defaults = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "beta": 0.0,
            "chi": 0.0,
            "psi": 0.0,
            "theta": 0.0,
            "alpha": np.deg2rad(90),
            "energy_notation": "Binding",
        }
        for k, v in defaults.items():
            data.attrs[k] = v
            for s in [dv for dv in data.data_vars.values() if "eV" in dv.dims]:
                s.attrs[k] = v

        data = data.rename({k: v for k, v in self.RENAME_KEYS.items() if k in data.coords})

        for coord in ["psi", "phi", "theta"]:
            if coord in data.coords:
                data = data.assign_coords({coord: np.deg2rad(data[coord])})

        return super().postprocess_final(data, scan_desc)


add_endstation(Phelix)
