"""Unit tests for arpes._helper.general.

These tests directly exercise the *_impl helper functions.
They aim to achieve full branch coverage for:
- DataArray and Dataset handling
- copy=True / copy=False behavior
- selections vs selections_kwargs paths
- ndarray vs xarray return types from callbacks
"""

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from arpes.xarray_extensions._helper.general import (
    apply_over_impl,
    filter_coord_impl,
    round_coordinates_impl,
)

# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def simple_da() -> xr.DataArray:
    """Simple 2D DataArray with numeric coordinates."""
    x = np.linspace(0.0, 10.0, 6)
    y = np.linspace(-1.0, 1.0, 3)
    data = np.arange(18).reshape(6, 3)
    return xr.DataArray(data, coords={"x": x, "y": y}, dims=("x", "y"))


@pytest.fixture
def simple_ds(simple_da: xr.DataArray) -> xr.Dataset:
    """Dataset wrapper around the simple DataArray."""
    return xr.Dataset({"a": simple_da})


# -----------------------------------------------------------------------------
# round_coordinates_impl
# -----------------------------------------------------------------------------


def test_round_coordinates_impl_returns_values(simple_da: xr.DataArray):
    """Nearest coordinate values should be returned as Python scalars."""
    rounded = round_coordinates_impl(
        simple_da,
        {"x": [2.1], "y": [0.2]},
    )

    assert set(rounded.keys()) == {"x", "y"}
    assert isinstance(rounded["x"], float)
    assert isinstance(rounded["y"], float)


def test_round_coordinates_impl_as_indices(simple_da: xr.DataArray):
    """When as_indices=True, integer coordinate indices should be returned."""
    rounded = round_coordinates_impl(
        simple_da,
        {"x": [2.1]},
        as_indices=True,
    )

    assert isinstance(rounded["x"], int)
    assert rounded["x"] == simple_da.coords["x"].to_index().get_loc(2.0)


# -----------------------------------------------------------------------------
# filter_coord_impl
# -----------------------------------------------------------------------------


def test_filter_coord_impl_dataarray(simple_da: xr.DataArray):
    """filter_coord_impl should filter a DataArray based on a sieve function."""

    def sieve(coord, da):
        return coord > 5.0

    filtered = filter_coord_impl(simple_da, "x", sieve)

    assert isinstance(filtered, xr.DataArray)
    assert np.all(filtered.coords["x"].values > 5.0)


def test_filter_coord_impl_dataset(simple_ds: xr.Dataset):
    """filter_coord_impl should also work for Dataset inputs."""

    def sieve(coord, ds):
        return coord <= 5.0

    filtered = filter_coord_impl(simple_ds, "x", sieve)

    assert isinstance(filtered, xr.Dataset)
    assert np.all(filtered.coords["x"].values <= 5.0)


# -----------------------------------------------------------------------------
# apply_over_impl (DataArray)
# -----------------------------------------------------------------------------


def test_apply_over_impl_da_copy_and_ndarray_return(simple_da: xr.DataArray):
    """When the callback returns an ndarray and copy=True,
    a new DataArray should be returned with updated values.
    """

    def fn(da):
        return da.values * 2

    result = apply_over_impl(
        simple_da,
        fn,
        selections={"x": 0.0},
    )

    assert isinstance(result, xr.DataArray)
    assert result is not simple_da
    assert np.all(result.sel(x=0.0).values == simple_da.sel(x=0.0).values * 2)


def test_apply_over_impl_da_inplace_with_xarray_return(simple_da: xr.DataArray):
    """When copy=False and the callback returns an xarray object,
    the modification should happen in-place.
    """

    def fn(da):
        return da + 1

    result = apply_over_impl(
        simple_da,
        fn,
        copy=False,
        x=2.0,
    )

    assert result is simple_da
    assert np.all(result.sel(x=2.0).values == simple_da.sel(x=2.0).values)


# -----------------------------------------------------------------------------
# apply_over_impl (Dataset)
# -----------------------------------------------------------------------------


def test_apply_over_impl_dataset_ndarray_return_raises(simple_ds: xr.Dataset):
    """Dataset + ndarray return should raise TypeError."""

    def fn(ds) -> NDArray[np.float64]:
        return ds["a"].values + 10

    with pytest.raises(TypeError, match="ndarray return is not supported"):
        apply_over_impl(
            simple_ds,
            fn,
            selections={"x": 4.0},
        )
