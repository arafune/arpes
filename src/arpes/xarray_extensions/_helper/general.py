"""Helper functions for xarray_extensions.general."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    overload,
)

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Mapping,
    )

    from numpy.typing import NDArray

    from arpes._typing.base import DataType, SelType


def round_coordinates_impl(
    data: DataType,
    coords_to_round: dict[str, list[float] | NDArray[np.float64]],
    *,
    as_indices: bool = False,
) -> dict[str, float | int]:
    """Shared implementation for round_coordinates.

    Assumes `data` is a concrete xarray object (DataArray or Dataset).
    """
    assert isinstance(data, xr.DataArray | xr.Dataset)
    rounded = {
        k: v.item()
        for k, v in data.sel(coords_to_round, method="nearest").coords.items()
        if k in coords_to_round
    }

    if as_indices:
        rounded = {k: data.coords[k].index(v) for k, v in rounded.items()}

    return rounded


@overload
def filter_coord_impl(
    data: xr.DataArray,
    coordinate_name: str,
    sieve: Callable[[Any, xr.DataArray], bool],
) -> xr.DataArray: ...


@overload
def filter_coord_impl(
    data: xr.Dataset,
    coordinate_name: str,
    sieve: Callable[[Any, xr.Dataset], bool],
) -> xr.Dataset: ...


def filter_coord_impl(
    data,
    coordinate_name,
    sieve,
):
    """Shared implementation for filter_coord.

    Assumes `data` is a concrete xarray object (DataArray or Dataset).
    """
    mask = np.array(
        [
            i
            for i, c in enumerate(data.coords[coordinate_name])
            if sieve(c, data.isel({coordinate_name: i}))
        ],
    )
    return data.isel({coordinate_name: mask})


@overload
def apply_over_impl(
    data: xr.Dataset,
    fn: Callable[[xr.Dataset], xr.Dataset | NDArray[np.float64]],
    *,
    copy: bool = True,
    selections: Mapping[str, SelType] | None = None,
    **selections_kwargs: SelType,
) -> xr.Dataset: ...


@overload
def apply_over_impl(
    data: xr.DataArray,
    fn: Callable[[xr.DataArray], xr.DataArray | NDArray[np.float64]],
    *,
    copy: bool = True,
    selections: Mapping[str, SelType] | None = None,
    **selections_kwargs: SelType,
) -> xr.DataArray: ...


def apply_over_impl(
    data: xr.DataArray | xr.Dataset,
    fn: Any,
    *,
    copy: bool = True,
    selections: Mapping[str, SelType] | None = None,
    **selections_kwargs: SelType,
) -> xr.DataArray | xr.Dataset:
    """Shared implementation for filter_coord.

    Assumes `data` is a concrete xarray object (DataArray or Dataset).
    """
    assert isinstance(data, xr.DataArray | xr.Dataset)

    data = data.copy(deep=True) if copy else data

    if selections is None:
        combined_selections: Mapping[str, SelType] = selections_kwargs
    else:
        combined_selections = {**(selections or {}), **selections_kwargs}

    selected: xr.DataArray | xr.Dataset = data.sel(**(combined_selections))  # type: ignore[arg-type]
    transformed = fn(selected)

    if isinstance(transformed, xr.DataArray | xr.Dataset):
        transformed = transformed.values

    data.loc[combined_selections] = transformed
    return data
