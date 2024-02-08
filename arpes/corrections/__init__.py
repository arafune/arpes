"""Provides standard corrections for datasets.

Largely, this covers:
1. Fermi edge corrections
2. Background estimation and subtraction

It also contains utilities related to identifying a piece of data
earlier in a dataset which can be used to furnish equivalent references.

"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import xarray as xr

from arpes.utilities import deep_equals, normalize_to_dataset

from .fermi_edge_corrections import (
    apply_direct_fermi_edge_correction,
    apply_photon_energy_fermi_edge_correction,
    apply_quadratic_fermi_edge_correction,
    build_direct_fermi_edge_correction,
    build_photon_energy_fermi_edge_correction,
    build_quadratic_fermi_edge_correction,
    find_e_fermi_linear_dos,
)

if TYPE_CHECKING:
    from arpes._typing import XrTypes

__all__ = (
    "reference_key",
    "correction_from_reference_set",
)


class HashableDict(OrderedDict):
    """Implements hashing for ordered dictionaries.

    The dictionary must be ordered for the hash to be stable.
    """

    def __hash__(self):
        return hash(frozenset(self.items()))


def reference_key(data: XrTypes) -> HashableDict:
    """Calculates a key/hash for data determining reference/correction equality."""
    data_array = normalize_to_dataset(data)
    assert isinstance(data_array, xr.DataArray)
    return HashableDict(data_array.S.reference_settings)


def correction_from_reference_set(data: XrTypes, reference_set):
    """Determines which correction to use from a set of references."""
    data_array = normalize_to_dataset(data)
    correction = None
    for k, corr in reference_set.items():
        if deep_equals(dict(reference_key(data_array)), dict(k)):
            correction = corr
            break

    return correction
