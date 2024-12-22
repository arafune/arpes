"""Defines common region selections used programmatically elsewhere."""

from __future__ import annotations

from enum import Enum
from typing import Literal

__all__ = ["REGIONS", "DesignatedRegions", "normalize_region"]


class DesignatedRegions(Enum):
    """Commonly used regions which can be used to select data programmatically."""

    # Angular windows
    NARROW_ANGLE = 0  # Narrow central region in the spectrometer
    WIDE_ANGLE = 1  # Just inside edges of spectremter data
    TRIM_EMPTY = 2  # Edges of spectrometer data

    # Energy windows
    BELOW_EF = 10  # Everything below e_F
    ABOVE_EF = 11  # Everything above e_F
    EF_NARROW = 12  # Narrow cut around e_F
    MESO_EF = 13  # Comfortably below e_F, pun on mesosphere

    # Effective energy windows, determined by Canny edge detection
    BELOW_EFFECTIVE_EF = 20  # Everything below e_F
    ABOVE_EFFECTIVE_EF = 21  # Everything above e_F
    EFFECTIVE_EF_NARROW = 22  # Narrow cut around e_F
    MESO_EFFECTIVE_EF = 23  # Comfortably below effective e_F, pun on mesosphere


REGIONS = {
    "copper_prior": {
        "eV": DesignatedRegions.MESO_EFFECTIVE_EF,
    },
    # angular can refer to either 'pixels' or 'phi'
    "wide_angular": {
        # angular can refer to either 'pixels' or 'phi'
        "angular": DesignatedRegions.WIDE_ANGLE,
    },
    "narrow_angular": {
        "angular": DesignatedRegions.NARROW_ANGLE,
    },
}


def normalize_region(
    region: Literal["copper_prior", "wide_angular", "narrow_angular"]
    | dict[str, DesignatedRegions],
) -> dict[str, DesignatedRegions]:
    """Converts named regions to an actual region."""
    if isinstance(region, str):
        return REGIONS[region]

    if isinstance(region, dict):
        return region

    msg = "Region should be either a string (i.e. an ID/alias) or an explicit dictionary."
    raise TypeError(
        msg,
    )


def find_spectrum_edge(data: xr.DataArray, *, axis_is_energy: bool = True, indices: bool = False):
    pass


def angle_selector(data: xr.DataArray, *, include_margin: bool = True):
    pass


def region_sel(data: xr.DataArray, regions) -> xr.DataArray:
    pass
