"""Provides a convenience data loader for example data.

Providing example data is essential for ensuring approachability,
but in the past we only provided a single ARPES cut. We now provide
a variety but need to be parsimonious about disk space for downloads.
As a result, this custom loader let's us pretend we store the data in
a higher quality format.
"""
from __future__ import annotations

from logging import DEBUG, INFO, Formatter, StreamHandler, getLogger
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import xarray as xr

import arpes.xarray_extensions  # noqa: F401
from arpes.endstations import SCANDESC, HemisphericalEndstation, SingleFileEndstation

if TYPE_CHECKING:
    from pathlib import Path

    from _typeshed import Incomplete

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


__all__ = ["ExampleDataEndstation"]


class ExampleDataEndstation(SingleFileEndstation, HemisphericalEndstation):
    """Loads data from exported .nc format saved by xarray. Used for storing example data."""

    PRINCIPAL_NAME = "example_data"

    _TOLERATED_EXTENSIONS: ClassVar[set[str]] = {".nc"}

    def load_single_frame(
        self,
        frame_path: str | Path = "",
        scan_desc: SCANDESC | None = None,
        **kwargs: Incomplete,
    ) -> xr.Dataset:
        """Loads single file examples.

        Additionally, copies coordinate offsets onto the dataset because we have
        preloaded these for convenience on maps.
        """
        if scan_desc:
            logger.debug("ExampleDataEndstation.load_single_frame: scan_desc is not used.")
        if kwargs:
            logger.debug("ExampleDataEndstation.load_single_frame: kwargs is not used.")
        data = xr.open_dataarray(frame_path)
        data = data.astype(np.float64)

        # Process coordinates so that there are no non-dimension coordinates
        # which are not a function of some index. This is for simplicity for beginners.
        replacement_coords = {}
        for cname, coord in data.coords.items():
            if len(coord.values.shape) and cname not in data.dims:
                replacement_coords[cname] = coord.mean().item()

        data = data.assign_coords(**replacement_coords)

        # Wrap into a dataset
        dataset = xr.Dataset({"spectrum": data})
        dataset.S.apply_offsets(data.S.offsets)

        return dataset
