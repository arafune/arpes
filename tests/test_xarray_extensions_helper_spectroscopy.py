"""Unit tests for _helper.spectroscopy.

These tests verify the behavior of sum_other_impl and mean_other_impl
for both xarray.DataArray and xarray.Dataset inputs.
"""

import numpy as np
import pytest
import xarray as xr

from arpes.xarray_extensions._helper.spectroscopy import mean_other_impl, sum_other_impl


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


# ---------------------------------------------------------------------
# sum_other_impl
# ---------------------------------------------------------------------


def test_sum_other_impl_dataarray(simple_da: xr.DataArray) -> None:
    """sum_other_impl should sum over all dimensions except the specified ones for a DataArray."""
    result = sum_other_impl(simple_da, ["x"])

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("x",)

    expected = simple_da.sum(dim="y")
    np.testing.assert_allclose(result.values, expected.values)


def test_sum_other_impl_dataset(simple_ds: xr.Dataset) -> None:
    """sum_other_impl should sum over all dimensions except the specified ones for a Dataset."""
    result = sum_other_impl(simple_ds, ["x"])

    assert isinstance(result, xr.Dataset)
    assert result["a"].dims == ("x",)

    expected = simple_ds.sum(dim="y")
    np.testing.assert_allclose(result["a"].values, expected["a"].values)


def test_sum_other_impl_keep_attrs(simple_da: xr.DataArray) -> None:
    """sum_other_impl should preserve attributes when keep_attrs=True."""
    result = sum_other_impl(simple_da, ["x"], keep_attrs=True)

    assert result.attrs == simple_da.attrs


# ---------------------------------------------------------------------
# mean_other_impl
# ---------------------------------------------------------------------


def test_mean_other_impl_dataarray(simple_da: xr.DataArray) -> None:
    """mean_other_impl should average over all dimensions except the specified ones for a DataArray."""
    result = mean_other_impl(simple_da, ["x"])

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("x",)

    expected = simple_da.mean(dim="y")
    np.testing.assert_allclose(result.values, expected.values)


def test_mean_other_impl_dataset(simple_ds: xr.Dataset):
    """mean_other_impl should average over all dimensions except the specified ones for a Dataset."""
    result = mean_other_impl(simple_ds, ["x"])

    assert isinstance(result, xr.Dataset)
    assert result["a"].dims == ("x",)

    expected = simple_ds.mean(dim="y")
    np.testing.assert_allclose(result["a"].values, expected["a"].values)


def test_mean_other_impl_keep_attrs(simple_da: xr.DataArray) -> None:
    """mean_other_impl should preserve attributes when keep_attrs=True."""
    result = mean_other_impl(simple_da, ["x"], keep_attrs=True)

    assert result.attrs == simple_da.attrs
