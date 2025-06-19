"""Implements loading the text file format for MB Scientific analyzers."""

from __future__ import annotations

import warnings
from logging import DEBUG, INFO
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr

from arpes.constants import TWO_DIMENSION
from arpes.debug import setup_logger
from arpes.endstations import HemisphericalEndstation
from arpes.utilities import clean_keys

if TYPE_CHECKING:
    from _typeshed import Incomplete

    from arpes._typing import ScanDesc

__all__ = ("MBSEndstation",)


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


class MBSEndstation(HemisphericalEndstation):
    """Implements loading text files from the MB Scientific text file format.

    There's not too much metadata here except what comes with the analyzer settings.
    """

    PRINCIPAL_NAME = "MBS"
    ALIASES: ClassVar[list[str]] = [
        "MB Scientific",
    ]
    _TOLERATED_EXTENSIONS: ClassVar[set[str]] = {
        ".txt",
    }

    RENAME_KEYS: ClassVar[dict[str, str]] = {
        "deflx": "psi",
    }

    def resolve_frame_locations(
        self,
        scan_desc: ScanDesc | None = None,
    ) -> list[Path | str]:
        """There is only a single file for the MBS loader, so this is simple."""
        if scan_desc is None:
            scan_desc = {}
        return [scan_desc.get("path", scan_desc.get("file"))]

    def postprocess_final(
        self,
        data: xr.Dataset,
        scan_desc: ScanDesc | None = None,
    ) -> xr.Dataset:
        """Performs final data normalization.

        Because the MBS format does not come from a proper ARPES DAQ setup,
        we have to attach a bunch of missing coordinates with blank values
        in order to fit the data model.
        """
        warnings.warn(
            "Loading from text format misses metadata. You will need to supply "
            "missing coordinates as appropriate.",
            stacklevel=2,
        )
        data.attrs["psi"] = float(data.attrs["psi"])
        for s in [dv for dv in data.data_vars.values() if "eV" in dv.dims]:
            s.attrs["psi"] = float(s.attrs["psi"])

        defaults = {
            "x": np.nan,
            "y": np.nan,
            "z": np.nan,
            "theta": 0,
            "beta": 0,
            "chi": 0,
            "alpha": np.nan,
            "hv": np.nan,
        }
        for k, v in defaults.items():
            data.attrs[k] = v
            for s in [dv for dv in data.data_vars.values() if "eV" in dv.dims]:
                s.attrs[k] = v

        return super().postprocess_final(data, scan_desc)

    def load_single_frame(
        self,
        frame_path: str | Path = "",
        scan_desc: ScanDesc | None = None,
        **kwargs: Incomplete,
    ) -> xr.Dataset:
        """Load a single frame from an MBS spectrometer.

        Most of the complexity here is in header handling and building
        the resultant coordinates. Namely, coordinates are stored in an indirect
        format using start/stop/step which needs to be hydrated.
        """
        if scan_desc:
            logger.debug("MBSEndstation.loadl_single_frame:scan_desc is not used")

        if kwargs:
            for k, v in kwargs.items():
                msg = "MBSEndstation.loadl_single_frame:"
                msg += f"unused kwargs is detected: k:{k}, v:{v}"
                logger.debug(msg)

        with Path(frame_path).open() as f:
            lines = f.readlines()

        lines = [line.strip() for line in lines]
        data_index = lines.index("DATA:")
        header = lines[:data_index]
        data = lines[data_index + 1 :]
        data_array = np.array([[float(f) for f in d] for d in [d.split() for d in data]])
        del data
        headers = [h.split("\t") for h in header]
        headers = [h for h in headers if len(h) == len(("item", "value"))]
        alt = [h for h in headers if len(h) == len(("only_item",))]
        headers.append(["alt", str(alt)])
        attrs = clean_keys(dict(headers))

        eV_axis = np.linspace(
            float(attrs["start_k_e_"]),
            float(attrs["end_k_e_"]),
            num=int(attrs["no_steps"]),
            endpoint=False,
        )

        n_eV = int(attrs["no_steps"])
        idx_eV = data_array.shape.index(n_eV)

        if data_array.ndim == TWO_DIMENSION:
            phi_axis = np.linspace(
                float(attrs["xscalemin"]),
                float(attrs["xscalemax"]),
                num=data_array.shape[1 if idx_eV == 0 else 0],
                endpoint=False,
            )

            coords = {"phi": np.deg2rad(phi_axis), "eV": eV_axis}
            dims = ["eV", "phi"] if idx_eV == 0 else ["phi", "eV"]
        else:
            coords = {"eV": eV_axis}
            dims = ["eV"]

        return xr.Dataset(
            {
                "spectrum": xr.DataArray(
                    data_array,
                    coords=coords,
                    dims=dims,
                    attrs=attrs,
                ),
            },
            attrs=attrs,
        )
