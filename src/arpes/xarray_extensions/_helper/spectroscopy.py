from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    overload,
)

if TYPE_CHECKING:
    import xarray as xr


@overload
def sum_other_impl(
    data: xr.Dataset,
    dim_or_dims: list[str],
    *,
    keep_attrs: bool = False,
) -> xr.Dataset: ...


@overload
def sum_other_impl(
    data: xr.DataArray,
    dim_or_dims: list[str],
    *,
    keep_attrs: bool = False,
) -> xr.DataArray: ...


def sum_other_impl(
    data: xr.Dataset | xr.DataArray,
    dim_or_dims: list[str],
    *,
    keep_attrs: bool = False,
):
    """Shared implementation for sum_other.

    Assumes `data` is a concrete xarray object (DataArray or Dataset).
    """
    assert isinstance(dim_or_dims, list)

    return data.sum(
        [d for d in data.dims if d not in dim_or_dims],
        keep_attrs=keep_attrs,
    )


@overload
def mean_other_impl(
    data: xr.Dataset,
    dim_or_dims: list[str] | str,
    *,
    keep_attrs: bool = False,
) -> xr.Dataset: ...


@overload
def mean_other_impl(
    data: xr.DataArray,
    dim_or_dims: list[str] | str,
    *,
    keep_attrs: bool = False,
) -> xr.DataArray: ...


def mean_other_impl(
    data: xr.Dataset | xr.DataArray,
    dim_or_dims: list[str] | str,
    *,
    keep_attrs: bool = False,
) -> xr.Dataset | xr.DataArray:
    """Shared implementation for mean_other.

    Assumes `data` is a concrete xarray object (DataArray or Dataset).
    """
    assert isinstance(dim_or_dims, list)

    return data.mean(
        [d for d in data.dims if d not in dim_or_dims],
        keep_attrs=keep_attrs,
    )
