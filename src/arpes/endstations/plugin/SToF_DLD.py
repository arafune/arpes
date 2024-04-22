"""Provides data loading for the Lanzara group experimental ARToF."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import h5py
import numpy as np
import xarray as xr

import arpes.config
from arpes.endstations import EndstationBase, ScanDesc
from arpes.provenance import Provenance, provenance_from_file

if TYPE_CHECKING:
    from _typeshed import Incomplete

    from arpes._typing import Spectrometer

__all__ = ("SToFDLDEndstation",)


class SToFDLDEndstation(EndstationBase):
    """Provides data loading for the Lanzara group experimental ARToF."""

    PRINCIPAL_NAME = "ALG-SToF-DLD"
    MERGE_ATTRS: ClassVar[Spectrometer] = {
        "length": 1.1456,  # This isn't correct but it should be a reasonable guess
    }

    def load(self, scan_desc: ScanDesc | None = None, **kwargs: Incomplete) -> xr.Dataset:
        """Load a FITS file containing run data from Ping and Anton's delay line detector ARToF.

        Params:
            scan_desc: Dictionary with extra information to attach to the xarray.Dataset,
            must contain the location of the file

        Returns:
            The loaded spectrum.
        """
        if scan_desc is None:
            warnings.warn(
                "Attempting to make due without user associated metadata for the file",
                stacklevel=2,
            )
            msg = "Expected a dictionary of metadata with the location of the file"
            raise TypeError(msg)
        if kwargs:
            warnings.warn("Any kwargs is not supported.", stacklevel=2)

        data_loc = Path(scan_desc["file"])
        if not data_loc.is_absolute():
            assert arpes.config.DATA_PATH is not None
            data_loc = Path(arpes.config.DATA_PATH) / data_loc

        f = h5py.File(data_loc, "r")

        dataset_contents = {}
        raw_data = f["/PRIMARY/DATA"][:]
        raw_data = raw_data[:, ::-1]  # Reverse the timing axis
        dataset_contents["raw"] = xr.DataArray(
            raw_data,
            coords={"x_pixels": np.linspace(0, 511, 512), "t_pixels": np.linspace(0, 511, 512)},
            dims=("x_pixels", "t_pixels"),
            attrs=f["/PRIMARY"].attrs.items(),
        )
        proenance_context: Provenance = {
            "what": "Loaded Anton and Ping DLD dataset from HDF5.",
            "by": "load_DLD",
        }

        provenance_from_file(dataset_contents["raw"], str(data_loc), proenance_context)

        return xr.Dataset(dataset_contents, attrs=scan_desc)
