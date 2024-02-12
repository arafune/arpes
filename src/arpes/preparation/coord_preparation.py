"""Utilities related to treating coordinates during data prep."""

from __future__ import annotations

import collections
import functools
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Sequence


__all__ = ["disambiguate_coordinates"]


def disambiguate_coordinates(
    datasets: xr.Dataset,
    possibly_clashing_coordinates: Sequence[str],
) -> list[xr.DataArray]:
    """Finds and unifies duplicated coordinates or ambiguous coordinates.

    This is useful if two regions claim to have an energy axis, but one is a core level
    and so refers to a different energy range.
    """
    coords_set = collections.defaultdict(list)
    for spectrum in datasets:
        assert isinstance(spectrum, xr.DataArray)
        for c in possibly_clashing_coordinates:
            if c in spectrum.coords:
                coords_set[c].append(spectrum.coords[c])

    conflicted = []
    for c in possibly_clashing_coordinates:
        different_coords = coords_set[c]
        if not different_coords:
            continue

        if not functools.reduce(
            lambda x, y: (np.array_equal(x[1], y) and x[0], y),
            different_coords,
            (True, different_coords[0]),
        )[0]:
            conflicted.append(c)

    after_deconflict = []
    for spectrum in datasets:
        assert isinstance(spectrum, xr.DataArray)
        spectrum_name = next(iter(spectrum.data_vars.keys()))
        to_rename = {
            name: str(name) + "-" + spectrum_name for name in spectrum.dims if name in conflicted
        }
        after_deconflict.append(spectrum.rename(to_rename))

    return after_deconflict
