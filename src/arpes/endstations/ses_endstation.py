"""Module for SES endstation data loading plugins.

This module contains classes and utilities for loading and processing data
from Scienta SESWrapper files and related endstation setups commonly used
in ARPES experiments.

Key classes:
- SESEndstation: Handles collation and loading of SESWrapper-format data,
  supporting both NetCDF (.nc) and PXT file formats.

The module provides file resolution, data loading, and coordinate normalization
tailored to SESWrapper data structures.
"""

from __future__ import annotations

from logging import DEBUG, INFO
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import xarray as xr

from arpes.configuration import get_config_manager
from arpes.debug import setup_logger
from arpes.load_pxt import find_ses_files_associated, read_single_pxt
from arpes.provenance import Provenance, provenance_from_file

from .base import EndstationBase
from .igor_utils import shim_wave_note

if TYPE_CHECKING:
    from arpes._typing.attrs_property import ScanDesc


LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


class SESEndstation(EndstationBase):
    """Provides collation and loading for Scienta's SESWrapper and endstations using it.

    These files have special frame names, at least at the beamlines Conrad has encountered.
    """

    def resolve_frame_locations(self, scan_desc: ScanDesc | None = None) -> list[Path]:
        if scan_desc is None:
            msg = "Must pass dictionary as file scan_desc to all endstation loading code."
            raise ValueError(
                msg,
            )

        original_data_loc = scan_desc.get("path", scan_desc.get("file"))
        assert original_data_loc
        config_manager = get_config_manager()
        if not Path(original_data_loc).exists():
            if config_manager.data_path is not None:
                original_data_loc = Path(config_manager.data_path) / original_data_loc
            else:
                msg = "File not found"
                raise RuntimeError(msg)

        return find_ses_files_associated(Path(original_data_loc))

    def load_single_frame(
        self,
        frame_path: str | Path = "",
        scan_desc: ScanDesc | None = None,
        **kwargs: bool,
    ) -> xr.Dataset:
        """Load the single frame from the specified file.

        This method loads a single frame of data from a file.
            If the file is in NetCDF (".nc") format, it loads the data using the `load_SES_nc`
            method, passing along the `scan_desc` dictionary and any additional keyword arguments.
            If the file is in PXT format, it reads the data, negates the energy values, and returns
            the data as an `xarray.Dataset` with the `spectrum` key.

        Args:
            frame_path (str | Path): The path to the file containing the single frame of data.
            scan_desc (ScanDesc | None): A description of the scan, which is passed to the
            `load_SES_nc` function if the file is in NetCDF format. Defaults to `None`.
            kwargs (bool): Additional keyword arguments passed to `load_SES_nc`. The only accepted
            argument is "robust_dimension_labels".

        Returns:
            xr.Dataset: The dataset containing the loaded spectrum data.
            Load the single frame from the file.
        """
        ext = Path(frame_path).suffix
        if scan_desc is None:
            scan_desc = {}
        if "nc" in ext:
            # was converted to hdf5/NetCDF format with Conrad's Igor scripts
            scan_desc["path"] = Path(frame_path)
            return self.load_SES_nc(scan_desc=scan_desc, **kwargs)

        # it's given by SES PXT files

        pxt_data = read_single_pxt(frame_path).assign_coords(
            {"eV": -read_single_pxt(frame_path).eV.values},
        )  # negate energy
        return xr.Dataset({"spectrum": pxt_data}, attrs=pxt_data.attrs)

    def postprocess(self, frame: xr.Dataset) -> xr.Dataset:
        frame = super().postprocess(frame)
        return frame.assign_attrs(frame.S.spectrum.attrs)

    def load_SES_nc(
        self,
        scan_desc: ScanDesc | None = None,
        *,
        robust_dimension_labels: bool = False,
    ) -> xr.Dataset:
        """Imports an hdf5 dataset exported from Igor that was originally generated in SES format.

        In order to understand the structure of these files have a look at Conrad's saveSESDataset
        in Igor Pro.

        Args:
            scan_desc: Dictionary with extra information to attach to the xr.Dataset, must contain
              the location of the file
            robust_dimension_labels: safety control, used to load despite possibly malformed
              dimension names
            kwargs: kwargs, unused currently

        Returns:
            Loaded data.
        """
        scan_desc = scan_desc or {}

        data_loc = scan_desc.get("path", scan_desc.get("file"))
        assert data_loc is not None
        config_manager = get_config_manager()
        if not Path(data_loc).exists():
            if config_manager.data_path is not None:
                data_loc = Path(config_manager.data_path) / data_loc
            else:
                msg = "File not found"
                raise RuntimeError(msg)

        wave_note = shim_wave_note(data_loc)
        f = h5py.File(data_loc, "r")

        primary_dataset_name = next(iter(f))
        # This is bugged for the moment in h5py due to an inability to read fixed length unicode
        # strings

        # Use dimension labels instead of
        dimension_labels = list(f["/" + primary_dataset_name].attrs["IGORWaveDimensionLabels"][0])
        if any(not x for x in dimension_labels):
            logger.info(dimension_labels)

            if not robust_dimension_labels:
                msg = "Missing dimension labels. Use robust_dimension_labels=True to override"
                raise ValueError(
                    msg,
                )
            used_blanks = 0
            for i in range(len(dimension_labels)):
                if not dimension_labels[i]:
                    dimension_labels[i] = f"missing{used_blanks}"
                    used_blanks += 1

            logger.info(dimension_labels)

        scaling = f["/" + primary_dataset_name].attrs["IGORWaveScaling"][-len(dimension_labels) :]
        raw_data = f["/" + primary_dataset_name][:]

        scaling = [
            np.linspace(
                scale[1],
                scale[1] + scale[0] * raw_data.shape[i],
                raw_data.shape[i],
                dtype=np.float64,
            )
            for i, scale in enumerate(scaling)
        ]

        dataset_contents = {}
        attrs = scan_desc.pop("note", {})
        attrs.update(wave_note)

        built_coords = dict(zip(dimension_labels, scaling, strict=True))

        deg_to_rad_coords = {"theta", "beta", "phi", "alpha", "psi"}

        # the hemisphere axis is handled below
        built_coords = {
            k: np.deg2rad(c) if k in deg_to_rad_coords else c for k, c in built_coords.items()
        }

        deg_to_rad_attrs = {"theta", "beta", "alpha", "psi", "chi"}
        for angle_attr in deg_to_rad_attrs:
            if angle_attr in attrs:
                attrs[angle_attr] = np.deg2rad(float(attrs[angle_attr]))

        dataset_contents["spectrum"] = xr.DataArray(
            raw_data,
            coords=built_coords,
            dims=dimension_labels,
            attrs=attrs,
        )
        provenance_context: Provenance = {"what": "Loaded SES dataset from HDF5.", "by": "load_SES"}
        provenance_from_file(dataset_contents["spectrum"], str(data_loc), provenance_context)
        return xr.Dataset(
            dataset_contents,
            attrs={**scan_desc, "dataset_name": primary_dataset_name},
        )
