"""Preliminary implementation of data loading at the ALS HERS beamline."""
from __future__ import annotations

import copy
import itertools
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr
from astropy.io import fits

import arpes.config
from arpes.endstations import HemisphericalEndstation, SynchrotronEndstation, find_clean_coords
from arpes.provenance import provenance_from_file
from arpes.utilities import rename_keys

if TYPE_CHECKING:
    from _typeshed import Incomplete

    from arpes.endstations import SCANDESC
__all__ = ("HERSEndstation",)


class HERSEndstation(SynchrotronEndstation, HemisphericalEndstation):
    """Implements data loading at the ALS HERS beamline.

    This should be unified with the FITs endstation code, but I don't have any projects at BL10
    at the moment so I will defer the complexity of unifying them for now
    """

    PRINCIPAL_NAME = "ALS-BL1001"
    ALIASES: ClassVar[list[str]] = ["ALS-BL1001", "HERS", "ALS-HERS", "BL1001"]

    def load(self, scan_desc: SCANDESC | None = None, **kwargs: Incomplete) -> xr.Dataset:
        """Loads HERS data from FITS files. Shares a lot in common with Lanzara group formats.

        Args:
            scan_desc: [TODO:description]
            kwargs: NOT Supported in this version.

        Raises:
            TypeError: [TODO:description]
        """
        if scan_desc is None:
            warnings.warn(
                "Attempting to make due without user associated scan_desc for the file",
                stacklevel=2,
            )
            msg = "Expected a dictionary of scan_desc with the location of the file"
            raise TypeError(msg)
        if kwargs:
            warnings.warn("Any kwargs is not supported in this function", stacklevel=2)

        scan_desc = dict(copy.deepcopy(scan_desc))

        data_loc = Path(scan_desc.get("path", scan_desc.get("file")))
        if not data_loc.is_absolute:
            data_loc = Path(arpes.config.DATA_PATH) / data_loc

        hdulist = fits.open(data_loc)

        hdulist[0].verify("fix+warn")
        _header_hdu, hdu = hdulist[0], hdulist[1]

        coords, dimensions, spectrum_shape = find_clean_coords(hdu, scan_desc)
        columns = hdu.columns  # pylint: disable=no-member

        column_renamings = {}
        take_columns = columns

        spectra_names = [name for name in take_columns if name in columns.names]

        skip_frags = {}
        skip_predicates = {lambda k: any(s in k for s in skip_frags)}
        scan_desc = {
            k: v for k, v in scan_desc.items() if not any(pred(k) for pred in skip_predicates)
        }

        data_vars = {
            k: (
                dimensions[k],
                hdu.data[k].reshape(spectrum_shape[k]),
                scan_desc,
            )  # pylint: disable=no-member
            for k in spectra_names
        }
        data_vars = rename_keys(data_vars, column_renamings)

        hdulist.close()

        relevant_dimensions = {
            k for k in coords if k in set(itertools.chain(*[_[0] for _ in data_vars.values()]))
        }
        relevant_coords = {k: v for k, v in coords.items() if k in relevant_dimensions}

        deg_to_rad_coords = {"beta", "psi", "chi", "theta"}
        relevant_coords = {
            k: c * (np.pi / 180) if k in deg_to_rad_coords else c
            for k, c in relevant_coords.items()
        }

        dataset = xr.Dataset(data_vars, relevant_coords, scan_desc)
        provenance_from_file(dataset, data_loc, {"what": "Loaded BL10 dataset", "by": "load_DLD"})

        return dataset
