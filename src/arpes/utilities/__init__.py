"""Provides general utility methods that get used during the course of analysis."""

from __future__ import annotations

import itertools
from operator import itemgetter
from typing import TYPE_CHECKING, Any

from .collections import deep_update
from .combine import concat_along_phi
from .dict import (
    clean_keys,
    rename_dataarray_attrs,
    rename_keys,
)
from .funcutils import Debounce, iter_leaves, lift_dataarray_to_generic
from .normalize import normalize_to_spectrum
from .region import REGIONS, DesignatedRegions, normalize_region
from .xarray import (
    apply_dataarray,
    lift_dataarray,
    lift_dataarray_attrs,
    lift_datavar_attrs,
    unwrap_xarray_dict,
    unwrap_xarray_item,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    import xarray as xr


def arrange_by_indices(items: list[Any], indices: list[int]) -> list[Any]:
    """Arranges `items` according to the new `indices` that each item should occupy.

    This function is best illustrated by the example below.
    It also has an inverse available in 'unarrange_by_indices'.

    Example:
        >>> arrange_by_indices(['a', 'b', 'c'], [1, 2, 0])
        ['b', 'c', 'a']
    """
    return [items[i] for i in indices]


def unarrange_by_indices(items: Sequence, indices: Sequence) -> list:
    """The inverse function to 'arrange_by_indices'.

    Ex:
    unarrange_by_indices(['b', 'c', 'a'], [1, 2, 0])
     => ['a', 'b', 'c']
    """
    return [x for x, _ in sorted(zip(indices, items, strict=True), key=itemgetter(0))]


ATTRS_MAP = {
    "PuPol": "pump_pol",
    "PrPol": "probe_pol",
    "SFLNM0": "lens_mode",
    "Lens Mode": "lens_mode",
    "Excitation Energy": "hv",
    "SFPE_0": "pass_energy",
    "Pass Energy": "pass_energy",
    "Slit Plate": "slit",
    "Number of Sweeps": "n_sweeps",
    "Acquisition Mode": "scan_mode",
    "Region Name": "scan_region",
    "Instrument": "instrument",
    "Pressure": "pressure",
    "User": "user",
    "Polar": "theta",
    "Theta": "theta",
    "Sample": "sample",
    "Beta": "beta",
    "Azimuth": "chi",
    "Location": "location",
}
