"""Unit test for axis_preparation.py."""

import numpy as np
import pytest
import xarray as xr

from arpes.preparation.axis_preparation import normalize_dim, vstack_data


def test_normalize_dim_single_dim():
    arr = xr.DataArray(
        np.array([[1, 2], [3, 4]]),
        dims=("x", "y"),
        coords={"x": [0, 1], "y": [0, 1]},
    )
    result = normalize_dim(arr, "x")
    assert np.isclose(result.mean().item(), 1.0)


def test_normalize_dim_multiple_dims():
    arr = xr.DataArray(
        np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
        dims=("x", "y", "z"),
        coords={"x": [0, 1], "y": [0, 1], "z": [0, 1]},
    )
    result = normalize_dim(arr, ["x", "y"])
    assert np.isclose(result.mean().item(), 2.0)


def test_normalize_dim_keep_id():
    arr = xr.DataArray(
        np.array([[1, 2], [3, 4]]),
        dims=("x", "y"),
        coords={"x": [0, 1], "y": [0, 1]},
        attrs={"id": "test_id"},
    )
    result = normalize_dim(arr, "x", keep_id=True)
    assert "id" in result.attrs and result.attrs["id"] == "test_id"


def test_normalize_dim_remove_id():
    arr = xr.DataArray(
        np.array([[1, 2], [3, 4]]),
        dims=("x", "y"),
        coords={"x": [0, 1], "y": [0, 1]},
        attrs={"id": "test_id"},
    )
    original_id = arr.attrs["id"]
    result = normalize_dim(arr, "x", keep_id=False)
    assert result.attrs["id"] != original_id


def test_vstack_data_valid_input():
    # Create mock DataArray objects with "new_dim" in attrs
    data1 = xr.DataArray([1, 2, 3], dims=["phi"], attrs={"new_dim": "A"})
    data2 = xr.DataArray([4, 5, 6], dims=["phi"], attrs={"new_dim": "B"})
    result = vstack_data([data1, data2], "new_dim")

    # Assert the concatenated result
    assert "new_dim" in result.coords
    assert list(result["new_dim"].values) == ["A", "B"]


def test_vstack_data_missing_new_dim_in_attrs():
    # Create mock DataArray objects with "new_dim" in coords
    data1 = xr.DataArray([1, 2, 3], dims=["phi"], coords={"new_dim": "A"})
    data2 = xr.DataArray([4, 5, 6], dims=["phi"], coords={"new_dim": "B"})
    result = vstack_data([data1, data2], "new_dim")

    # Assert the concatenated result
    assert "new_dim" in result.coords
    assert list(result["new_dim"].values) == ["A", "B"]


def test_vstack_data_missing_new_dim_in_both():
    # Create mock DataArray objects without "new_dim" in attrs or coords
    data1 = xr.DataArray([1, 2, 3], dims=["phi"])
    data2 = xr.DataArray([4, 5, 6], dims=["phi"])

    with pytest.raises(AssertionError):
        vstack_data([data1, data2], "new_dim")


def test_vstack_data_remove_new_dim_from_attrs():
    # Create mock DataArray objects with "new_dim" in attrs
    data1 = xr.DataArray([1, 2, 3], dims=["phi"], attrs={"new_dim": "A"})
    data2 = xr.DataArray([4, 5, 6], dims=["phi"], attrs={"new_dim": "B"})
    result = vstack_data([data1, data2], "new_dim")
    # Assert "new_dim" is removed from attrs
    assert "new_dim" not in result.attrs
