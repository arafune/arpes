"""Unit tests for xarray_extensions/accessor/spectroscopy.py."""

import numpy as np
import pytest
import xarray as xr

import arpes.xarray_extensions  # noqa: F401


@pytest.fixture
def simple_da() -> xr.DataArray:
    """Simple 2D DataArray for testing."""
    x = [0.0, 1.0, 2.0]
    y = [10.0, 20.0]
    data = np.arange(6).reshape(3, 2)
    return xr.DataArray(
        data,
        coords={"x": x, "y": y},
        dims=("x", "y"),
        name="a",
        attrs={"test_attr": "value"},
    )


@pytest.fixture
def simple_ds(simple_da: xr.DataArray) -> xr.Dataset:
    """Simple Dataset wrapping simple_da."""
    return xr.Dataset({"a": simple_da})


def test_sum_other_for_da(simple_da: xr.DataArray) -> None:
    """Test for thin wrapper of sum_other_impl for DataArray."""
    result = simple_da.S.sum_other(["x"])

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("x",)

    expected = simple_da.sum(dim="y")
    np.testing.assert_allclose(result.values, expected.values)


def test_mean_other_for_da(simple_da: xr.DataArray) -> None:
    """Test for thin wrapper of mean_other_impl for DataArray."""
    result = simple_da.S.mean_other(["x"])

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("x",)

    expected = simple_da.mean(dim="y")
    np.testing.assert_allclose(result.values, expected.values)


def test_sum_other_for_ds(simple_ds: xr.Dataset) -> None:
    """Test for thin wrapper of sum_other_impl for Dataset."""
    result = simple_ds.S.sum_other(["x"])

    assert isinstance(result, xr.Dataset)
    assert result["a"].dims == ("x",)

    expected = simple_ds.sum(dim="y")
    np.testing.assert_allclose(result["a"].values, expected["a"].values)


def test_mean_other_for_ds(simple_ds: xr.Dataset) -> None:
    """Test for thin wrapper of mean_other_impl for Dataset."""
    result = simple_ds.S.mean_other(["x"])

    assert isinstance(result, xr.Dataset)
    assert result["a"].dims == ("x",)

    expected = simple_ds.mean(dim="y")
    np.testing.assert_allclose(result["a"].values, expected["a"].values)
