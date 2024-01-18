"""Implements loading the text file format for MB Scientific analyzers."""
from __future__ import annotations

import warnings
from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr

from arpes.endstations import SCANDESC, HemisphericalEndstation
from arpes.utilities import clean_keys

if TYPE_CHECKING:
    from _typeshed import Incomplete

__all__ = ("MBSEndstation",)


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = getLogger(__name__)
fmt = "%(asctime)s %(levelname)s %(name)s :%(message)s"
formatter = Formatter(fmt)
handler = StreamHandler()
handler.setLevel(LOGLEVEL)
logger.setLevel(LOGLEVEL)
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


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
        scan_desc: SCANDESC | None = None,
    ) -> list[str | Path]:
        """There is only a single file for the MBS loader, so this is simple."""
        return [scan_desc.get("path", scan_desc.get("file"))]

    def postprocess_final(
        self,
        data: xr.Dataset,
        scan_desc: SCANDESC | None = None,
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
        for s in data.S.spectra:
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
            for s in data.S.spectra:
                s.attrs[k] = v

        return super().postprocess_final(data, scan_desc)

    def load_single_frame(
        self,
        frame_path: str | None = None,
        scan_desc: SCANDESC | None = None,
        **kwargs: Incomplete,
    ) -> xr.Dataset:
        """Load a single frame from an MBS spectrometer.

        Most of the complexity here is in header handling and building
        the resultant coordinates. Namely, coordinates are stored in an indirect
        format using start/stop/step which needs to be hydrated.
        """
        if scan_desc:
            for k, v in scan_desc:
                logger.debug(f"scan_desc is not used: k{k}, v{v}")

        if kwargs:
            for k, v in kwargs:
                logger.debug(f"unused kwargs is detected: k:{k}, v:{v}")

        with Path(frame_path).open() as f:
            lines = f.readlines()

        lines = [_.strip() for _ in lines]
        data_index = lines.index("DATA:")
        header = lines[:data_index]
        data = lines[data_index + 1 :]
        data = [d.split() for d in data]
        data = np.array([[float(f) for f in d] for d in data])

        header = [h.split("\t") for h in header]
        alt = [h for h in header if len(h) == 1]
        header = [h for h in header if len(h) == 2]
        header.append(["alt", str(alt)])
        attrs = clean_keys(dict(header))

        eV_axis = np.linspace(
            float(attrs["start_k_e_"]),
            float(attrs["end_k_e_"]),
            num=int(attrs["no_steps"]),
            endpoint=False,
        )

        n_eV = int(attrs["no_steps"])
        idx_eV = data.shape.index(n_eV)

        if len(data.shape) == 2:
            phi_axis = np.linspace(
                float(attrs["xscalemin"]),
                float(attrs["xscalemax"]),
                num=data.shape[1 if idx_eV == 0 else 0],
                endpoint=False,
            )

            coords = {"phi": phi_axis * np.pi / 180, "eV": eV_axis}
            dims = ["eV", "phi"] if idx_eV == 0 else ["phi", "eV"]
        else:
            coords = {"eV": eV_axis}
            dims = ["eV"]

        return xr.Dataset(
            {
                "spectrum": xr.DataArray(
                    data,
                    coords=coords,
                    dims=dims,
                    attrs=attrs,
                ),
            },
            attrs=attrs,
        )
