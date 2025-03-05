"""Provides general utility methods that get used during the course of analysis."""

# pyright: reportUnusedImport=false
from __future__ import annotations

import warnings
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
from .math import (
    fermi_distribution,
    inv_fermi_distribution,
    polarization,
)
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


def arrange_by_indices(items: list[Any], indices: list[int]) -> list[Any]:  # pragma: no cover
    """Arranges `items` according to the new `indices` that each item should occupy.

    This function is best illustrated by the example below.
    It also has an inverse available in 'unarrange_by_indices'.

    Example:
        >>> arrange_by_indices(['a', 'b', 'c'], [1, 2, 0])
        ['b', 'c', 'a']
    """
    warnings.warn(
        "This method will be deprecated.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return [items[i] for i in indices]


def unarrange_by_indices(items: Sequence, indices: Sequence) -> list:  # pragma: no cover
    """The inverse function to 'arrange_by_indices'.

    Examples:
    unarrange_by_indices(['b', 'c', 'a'], [1, 2, 0])
     => ['a', 'b', 'c']
    """
    warnings.warn(
        "This method will be deprecated.",
        category=DeprecationWarning,
        stacklevel=2,
    )
    return [x for x, _ in sorted(zip(indices, items, strict=True), key=itemgetter(0))]
