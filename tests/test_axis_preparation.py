"""Unit test for axis_preparation.py."""

import numpy as np
import pytest
import xarray as xr

from arpes.preparation.axis_preparation import normalize_dim


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
