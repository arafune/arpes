"""Implements data loading for the IF UMCS Lublin ARPES group."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr

from arpes.endstations import (
    SCANDESC,
    HemisphericalEndstation,
    SingleFileEndstation,
    add_endstation,
)
from arpes.endstations.prodigy_xy import load_xy

if TYPE_CHECKING:
    from arpes._typing import SPECTROMETER
    from arpes.endstations import SCANDESC

__all__ = ("IF_UMCS",)


class IF_UMCS(HemisphericalEndstation, SingleFileEndstation):
    """Implements loading xy text files from the Specs Prodigy software."""

    PRINCIPAL_NAME = "IF_UMCS"
    ALIASES: ClassVar = ["IF_UMCS", "LubARPES", "LublinARPRES"]

    _TOLERATED_EXTENSIONS: ClassVar = {".xy"}

    RENAME_KEYS: ClassVar = {
        "eff_workfunction": "workfunction",
        "analyzer_slit": "slit",
        "analyzer_lens": "lens_mode",
        "detector_voltage": "mcp_voltage",
    }

    MERGE_ATTRS: ClassVar[SPECTROMETER] = {
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
            scan_desc: SCANDESC | None = None,
            **kwargs: str | float,
    ) -> xr.Dataset:
        """Load single xy file."""
        if scan_desc is None:
            scan_desc = {}
        file = Path(frame_path)
        if file.suffix == ".xy":
            data = load_xy(frame_path, **kwargs)
            if not isinstance(data, list):
                return xr.Dataset({"spectrum": data}, attrs=data.attrs)

        msg = "Data file must be ended with .xy"
        raise RuntimeError(msg)

    def postprocess_final(
            self,
            data: xr.Dataset,
            scan_desc: SCANDESC | None = None,
    ) -> xr.Dataset:
        """Add missing parameters."""
        if scan_desc is None:
            scan_desc = {}
        defaults = {
            "x": 78,
            "y": 0.5,
            "z": 2.5,
            "theta": 0,
            "beta": 0,
            "chi": 0,
            "psi": 0,
            "alpha": np.deg2rad(90),
            "hv": 21.2,
        }
        for k, v in defaults.items():
            data.attrs[k] = v
            for s in data.S.spectra:
                s.attrs[k] = v

        return super().postprocess_final(data, scan_desc)


add_endstation(IF_UMCS)
