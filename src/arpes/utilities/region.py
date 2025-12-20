"""Defines common region selections used programmatically elsewhere."""

from __future__ import annotations

from enum import Enum
from logging import DEBUG, INFO
from typing import TYPE_CHECKING, Literal

import numpy as np

from arpes.debug import setup_logger

if TYPE_CHECKING:
    from collections.abc import Callable

    import xarray as xr

    from arpes._typing.base import AnalysisRegion

__all__ = ["REGIONS", "DesignatedRegions", "normalize_region"]

LOGLEVELS = (DEBUG, INFO)
LOGLEVEL = LOGLEVELS[1]
logger = setup_logger(__name__, LOGLEVEL)


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


def wide_angle_selector(
    data: xr.DataArray,
    *,
    include_margin: bool = True,
) -> slice:
    """Generates a slice for selecting the wide angular range of the spectrum.

    Optionally includes a margin to slightly reduce the range.

    Args:
        data (xr.DataArray): The spectrum data.
        include_margin (bool, optional): If True, includes a margin to shrink the range.
            Defaults to True.

    Returns:
        slice: A slice object representing the wide angular range of the spectrum.

    Todo:
        - Add tests.
        - Consider removing the function.

    """
    # Too difficult to avoid circular import w/o using lazy import
    from arpes.analysis.spectrum_edges import find_spectrum_angular_edges  # noqa: PLC0415

    edges = find_spectrum_angular_edges(data)
    low_edge, high_edge = np.min(edges), np.max(edges)

    # go and build in a small margin
    if include_margin:
        if "pixels" in data.dims:
            low_edge += 50
            high_edge -= 50
        else:
            low_edge += 0.05
            high_edge -= 0.05

    return slice(low_edge, high_edge)


def meso_effective_selector(data: xr.DataArray) -> slice:
    """Creates a slice to select the "meso-effective" range of the spectrum.

    The range is defined as the upper energy range from `max(energy_edge) - 0.3` to
    `max(energy_edge) - 0.1`.

    Args:
        data (xr.DataArray): The spectrum data.

    Returns:
        slice: A slice object representing the meso-effective energy range.

    Todo:
        - Add tests.
        - Consider removing the function.

    """
    # Too difficult to avoid circular import w/o using lazy import
    from arpes.analysis.spectrum_edges import find_spectrum_energy_edges  # noqa: PLC0415

    energy_edge = find_spectrum_energy_edges(data)
    return slice(energy_edge.max().item() - 0.3, energy_edge.max().item() - 0.1)


def region_sel(
    data: xr.DataArray,
    *regions: AnalysisRegion | dict[str, DesignatedRegions],
) -> xr.DataArray:
    """Filters the data by selecting specified regions and applying those regions to the object.

    Regions can be provided as literal strings or as a dictionary of `DesignatedRegions`.

    Args:
        data (xr.DataArray): The data to filter.
        regions (Literal or dict[str, DesignatedRegions]): The regions to select.
            Valid regions include:
            - "copper_prior": A specific region.
            - "wide_angular": The wide angular region.
            - "narrow_angular": The narrow angular region.
            Alternatively, use the `DesignatedRegions` enumeration.

    Returns:
        xr.DataArray: The data with the selected regions applied.

    Raises:
        NotImplementedError: If a specified region cannot be resolved.

    Todo:
        - Add tests.
    """

    def process_region_selector(
        selector: slice | DesignatedRegions,
        dimension_name: str,
    ) -> slice | Callable[..., slice]:
        if isinstance(selector, slice):
            return selector

        options = {
            "eV": (
                DesignatedRegions.ABOVE_EF,
                DesignatedRegions.BELOW_EF,
                DesignatedRegions.EF_NARROW,
                DesignatedRegions.MESO_EF,
                DesignatedRegions.MESO_EFFECTIVE_EF,
                DesignatedRegions.ABOVE_EFFECTIVE_EF,
                DesignatedRegions.BELOW_EFFECTIVE_EF,
                DesignatedRegions.EFFECTIVE_EF_NARROW,
            ),
            "phi": (
                DesignatedRegions.NARROW_ANGLE,
                DesignatedRegions.WIDE_ANGLE,
                DesignatedRegions.TRIM_EMPTY,
            ),
        }

        options_for_dim = options.get(dimension_name, list(DesignatedRegions))
        assert selector in options_for_dim

        # now we need to resolve out the region
        resolution_methods = {
            DesignatedRegions.ABOVE_EF: slice(0, None),
            DesignatedRegions.BELOW_EF: slice(None, 0),
            DesignatedRegions.EF_NARROW: slice(-0.1, 0.1),
            DesignatedRegions.MESO_EF: slice(-0.3, -0.1),
            DesignatedRegions.MESO_EFFECTIVE_EF: meso_effective_selector(data),
            # Implement me
            # DesignatedRegions.TRIM_EMPTY: ,
            DesignatedRegions.WIDE_ANGLE: wide_angle_selector(data),
            # DesignatedRegions.NARROW_ANGLE: self.narrow_angle_selector,
        }
        resolution_method = resolution_methods[selector]
        if isinstance(resolution_method, slice):
            return resolution_method
        if callable(resolution_method):
            return resolution_method()

        msg = "Unable to determine resolution method."
        raise NotImplementedError(msg)

    obj = data

    def unpack_dim(dim_name: str) -> str:
        if dim_name == "angular":
            return "pixel" if "pixel" in obj.dims else "phi"

        return dim_name

    for region in regions:
        # remove missing dimensions from selection for permissiveness
        # and to transparent composing of regions
        obj = obj.sel(
            {
                k: process_region_selector(v, k)
                for k, v in {unpack_dim(k): v for k, v in normalize_region(region).items()}.items()
                if k in obj.dims
            },
        )

    return obj
