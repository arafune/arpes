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
