"""Implements data loading for the IF UMCS Lublin ARPES group."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr

from arpes.endstations import (
    HemisphericalEndstation,
    ScanDesc,
    SingleFileEndstation,
    add_endstation,
)
from arpes.endstations.prodigy_xy import load_xy

if TYPE_CHECKING:
    from arpes._typing import Spectrometer
    from arpes.endstations import ScanDesc

__all__ = ["IF_UMCSEndstation"]


class IF_UMCSEndstation(  # noqa: N801
    HemisphericalEndstation,
    SingleFileEndstation,
):
    """Implements loading xy text files from the Specs Prodigy software."""

    PRINCIPAL_NAME = "IF_UMCS"
    ALIASES: ClassVar[list[str]] = ["IF_UMCS", "LubARPES", "LublinARPRES"]

    _TOLERATED_EXTENSIONS: ClassVar[set[str]] = {".xy", ".itx"}

    LENS_MAPPING: ClassVar[dict[str, bool]] = {
        "HighAngularDispersion": True,
        "MediumAngularDispersion": True,
        "LowAngularDispersion":  True,
        "WideAngleMode": True,
        "LowMagnification": False,
        "MediumMagnification": False,
        "HighMagnification": False,
    }

    RENAME_KEYS: ClassVar[dict[str, str]] = {
        "eff_workfunction": "workfunction",
        "analyzer_slit": "slit",
        "analyzer_lens": "lens_mode",
        "detector_voltage": "mcp_voltage",
        "excitation_energy": "hv",
        "polar": "theta",
        "region": "id",
    }

    MERGE_ATTRS: ClassVar[Spectrometer] = {
        "analyzer": "Specs PHOIBOS 150",
        "analyzer_name": "Specs PHOIBOS 150",
        "parallel_deflectors": False,
        "perpendicular_deflectors": False,
        "analyzer_radius": 150,
        "analyzer_type": "hemispherical",
    }

    def load_single_frame(
        self,
        frame_path: str | Path = "",
        scan_desc: ScanDesc | None = None,
        **kwargs: str | float,
    ) -> xr.Dataset:
        """Load single file."""
        if scan_desc is None:
            scan_desc = {}
        file = Path(frame_path)
        if file.suffix in self._TOLERATED_EXTENSIONS:
            if file.suffix == ".xy":
                data = load_xy(frame_path, **kwargs)
            elif file.suffix == ".itx":
                msg = "Not suported yet..."
                raise RuntimeWarning(msg)
            return xr.Dataset({"spectrum": data}, attrs=data.attrs)

        msg = "Data file must be ended with .xy or .itx"
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
        is the theta angle in this ARPES apparatus.

        Args:
            data(xr.Dataset): ARPES data
            scan_desc(SCANDESC | None): scan_description. Not used currently

        Returns:
            xr.Dataset: pyARPES compatible.
        """
        # Convert to binding energy
        binding_energies = data.coords["eV"].values - data.attrs["hv"]
        data = data.assign_coords({"eV": binding_energies})
        lens_mode = data.attrs["lens_mode"].split(":")[0]

        if lens_mode in self.LENS_MAPPING:
            dispersion_mode = self.LENS_MAPPING[lens_mode]
            if dispersion_mode:
                data = data.rename({"nonenergy": "phi"})
                data = data.assign_coords(phi=np.deg2rad(data.phi))
            else:
                data = data.rename({"nonenergy": "x"})
        else:
            msg = f"Unknown Analyzer Lens: {lens_mode}"
            raise ValueError(msg)

        """Add missing parameters."""
        if scan_desc is None:
            scan_desc = {}
        defaults = {
            "x": 78,
            "y": 0.5,
            "z": 2.5,
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
        if "theta" in data.coords:
            data = data.assign_coords(theta=np.deg2rad(data.theta))

        return super().postprocess_final(data, scan_desc)


add_endstation(IF_UMCSEndstation)
