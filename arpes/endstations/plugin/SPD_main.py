"""Implements loading the itx and sp2 text file format for SPECS prodigy."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr

from arpes.endstations import (
    HemisphericalEndstation,
    SingleFileEndstation,
    add_endstation,
)
from arpes.endstations.prodigy_itx import load_itx, load_sp2

if TYPE_CHECKING:
    from arpes._typing import SPECTROMETER

__all__ = [
    "SPDEndstation",
]


class SPDEndstation(HemisphericalEndstation, SingleFileEndstation):
    """Class for PHOIBOS 100 controlled by using prodigy."""

    PRINCIPAL_NAME = "SPD"
    ALIASES: ClassVar[list[str]] = [
        "SPD_phoibos",
    ]
    _TOLERATED_EXTENSIONS: ClassVar[set[str]] = {".itx", ".sp2"}

    RENAME_KEYS: ClassVar[dict[str, str]] = {
        "Excitation Energy": "hv",
        "WorkFunction": "workfunction",
        # Workfunction of ANALYZER (Don't confuse sample_workfunction)
        "WF": "workfunction",
        "Lens Mode": "lens_mode",
        "lensmode": "lens_mode",
        "Pass Energy": "pass_energy",
        "Pass Energy (Target) [eV]": "pass_energy",
        "DetectorVoltage [V]": "mcp_voltage",
        "Detector Voltage": "mcp_voltage",
        "Spectrum ID": "id",
    }

    MERGE_ATTRS: ClassVar[SPECTROMETER] = {
        "analyzer": "Specs PHOIBOS 100",
        "analyzer_name": "Specs PHOIBOS 100",
        "parallel_deflectors": False,
        "perpendicular_deflectors": False,
        "analyzer_radius": 100,
        "analyzer_type": "hemispherical",
        #
        "alpha": np.pi / 2,
        "chi": 0,
        "theta": 0,
        "psi": 0,
    }

    def postprocess_final(
        self,
        data: xr.Dataset,
        scan_desc: dict[str, str] | None = None,
    ) -> xr.Dataset:
        """Perform final data normalization.

        Args:
            data(xr.Dataset): _description_
            scan_desc(dict:[str, str]| None): _description_, by default None

        Returns:
            xr.Dataset: pyARPES compatible.
        """
        if scan_desc is None:
            scan_desc = {}
        defaults = {
            "x": np.nan,
            "y": np.nan,
            "z": np.nan,
            "theta": 0,
            "beta": 0,
            "chi": 0,
            "alpha": np.pi / 2,
            "hv": np.nan,
            "energy_notation": "Kinetic",
        }
        for k, v in defaults.items():
            data.attrs[k] = data.attrs.get(k, v)
            for s in data.S.spectra:
                s.attrs[k] = s.attrs.get(k, v)
        return super().postprocess_final(data, scan_desc)

    def load_single_frame(
        self,
        frame_path: str = "",
        scan_desc: dict[str, str] | None = None,
        **kwargs: str | int | float,
    ) -> xr.Dataset:
        """Load a single frame from an PHOIBOS 100 spectrometer with Prodigy.

        Args:
            frame_path(str): _description_, by default ""
            scan_desc(dict[str, str] | None): _description_, by default None
            kwargs(str | int | float): Pass to load_itx

        Returns:
            xr.Datast: pyARPES is not compatible at this stage.  (postprocess_final is needed.)
        """
        if scan_desc is None:
            scan_desc = {}
        file = Path(frame_path)
        if file.suffix == ".itx":
            data = load_itx(frame_path, **kwargs)
            if not isinstance(data, list):
                return xr.Dataset({"spectrum": data}, attrs=data.attrs)
            if not is_dim_coords_same_all(data):
                msg = "Dimension (coordinate) mismatch"
                raise RuntimeError(msg)
            return xr.Dataset(
                {"ID_" + str(an_array.attrs["id"]).zfill(3): an_array for an_array in data},
            )
        if file.suffix == ".sp2":
            data = load_sp2(frame_path, **kwargs)
            return xr.Dataset({"spectrum": data}, attrs=data.attrs)
        msg = "Data file must be ended with .itx or .sp2"
        raise RuntimeError(msg)


def is_dim_coords_same(a: xr.DataArray, b: xr.DataArray) -> bool:
    """Returns true if the coords used in dims are same in two DataArray."""
    assert isinstance(a, xr.DataArray)
    assert isinstance(b, xr.DataArray)
    try:
        return all(np.array_equal(a.coords[dim], b.coords[dim]) for dim in a.dims)
    except KeyError:
        return False


def is_dim_coords_same_all(list_of_dataarrays: list[xr.DataArray]) -> bool:
    for i in range(len(list_of_dataarrays)):
        if not is_dim_coords_same(list_of_dataarrays[i - 1], list_of_dataarrays[i]):
            return False
    return True


add_endstation(SPDEndstation)