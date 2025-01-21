"""Implements data loading for the ARPES at DSNP UMCS.

(Department of Surface and Nanostructures Physics)
Institute of Physics, Maria Curie-Sklodowska University in Lublin.
"""

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
from arpes.provenance import Provenance, provenance_from_file

if TYPE_CHECKING:
    from collections.abc import Callable

    from arpes._typing import Spectrometer
    from arpes.endstations import ScanDesc


__all__ = ["DSNP_UMCSEndstation"]


class DSNP_UMCSEndstation(  # noqa: N801
    HemisphericalEndstation,
    SingleFileEndstation,
):
    """Implements loading xy text files from the Specs Prodigy software."""

    PRINCIPAL_NAME = "DSNP_UMCS"
    ALIASES: ClassVar[list[str]] = ["DSNP_UMCS", "IF_UMCS", "LubARPES", "LublinARPRES"]

    _TOLERATED_EXTENSIONS: ClassVar[set[str]] = {".xy", ".itx"}

    # Mapping of lens modes to dispersion or magnification modes
    _LENS_MAPPING: ClassVar[dict[str, bool]] = {
        "HighAngularDispersion": True,
        "MediumAngularDispersion": True,
        "LowAngularDispersion": True,
        "WideAngleMode": True,
        "LowMagnification": False,
        "MediumMagnification": False,
        "HighMagnification": False,
    }

    RENAME_KEYS: ClassVar[dict[str, str]] = {
        "eff_workfunction": "workfunction",
        "acquisition_date": "aquisition_date_utc",
        "analyzer_slit": "slit",
        "analyzer_lens": "lens_mode",
        "detector_voltage": "mcp_voltage",
        "excitation_energy": "hv",
        "region": "id",
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
        """Load single frame from Specs Prodiy generated xy file.

        - Calculate phi or x values depending on the lens mode.
        - Rename the third dimension (polar) to theta angle.
        - Convert degrees to radians.

        Args:
            frame_path(str | Path): _description_, by default ""
            scan_desc(ScanDesc | None): _description_, by default None
            kwargs(str | int | float): Pass to load_xy

        Returns:
            xr.Datast: pyARPES is not compatible at this stage.  (postprocess_final is needed.)
        """
        provenance_context: Provenance = {
            "what": "Loaded xy dataset",
            "by": "load_single_frame",
        }

        if scan_desc is None:
            scan_desc = {}
        file = Path(frame_path)
        if file.suffix in self._TOLERATED_EXTENSIONS:
            if file.suffix == ".xy":
                data = load_xy(frame_path, **kwargs)

                # Calculate phi or x values depending on the lens mode
                lens_mode = data.attrs["analyzer_lens"].split(":")[0]
                if lens_mode in self._LENS_MAPPING:
                    dispersion_mode = self._LENS_MAPPING[lens_mode]
                    if dispersion_mode:
                        data = data.rename({"nonenergy": "phi"})
                        data = data.assign_coords(phi=np.deg2rad(data.phi))
                    else:
                        data = data.rename({"nonenergy": "x"})
                else:
                    msg = f"Unknown Analyzer Lens: {lens_mode}"
                    raise ValueError(msg)

                # Convert polar manipulator angle to theta in radians
                if "polar" in data.coords:
                    data = data.rename({"polar": "theta"})
                    data = data.assign_coords(theta=np.deg2rad(data.theta))

                dataset = xr.Dataset({"spectrum": data}, attrs=data.attrs)
                provenance_from_file(
                    child_arr=dataset["spectrum"],
                    file=str(frame_path),
                    record=provenance_context,
                )
                dataset.attrs["location"] = self.PRINCIPAL_NAME
                return dataset

            if file.suffix == ".itx":
                msg = "Not supported yet..."
                raise RuntimeError(msg)

        msg = "Data file must be ended with .xy or .itx"
        raise RuntimeError(msg)

    def postprocess_final(
        self,
        data: xr.Dataset,
        scan_desc: ScanDesc | None = None,
    ) -> xr.Dataset:
        """Perform final processing on the ARPES data.

        - Add missing parameters.

        Args:
            data(xr.Dataset): ARPES data
            scan_desc(SCANDESC | None): scan_description. Not used currently

        Returns:
            xr.Dataset: pyARPES compatible.
        """
        # Convert to binding energy
        binding_energies = data.coords["eV"].values - data.attrs["hv"]
        data = data.assign_coords({"eV": binding_energies})

        # Add missing parameters
        if scan_desc is None:
            scan_desc = {}
        defaults = {
            "x": 78.0,
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

        data = super().postprocess_final(data, scan_desc)
        data.S.spectrum.attrs["location"] = self.PRINCIPAL_NAME
        return data


add_endstation(DSNP_UMCSEndstation)
