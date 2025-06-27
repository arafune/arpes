import numpy as np
import pytest
import xarray as xr

from src.arpes.analysis.filters import boxcar_filter_arr


@pytest.fixture
def sample_data():
    coords = {"x": np.linspace(0, 10, 11), "y": np.linspace(0, 20, 21)}
    data = np.random.random((11, 21))
    return xr.DataArray(data, coords=coords, dims=["x", "y"])


def test_boxcar_filter_arr_with_pixel_units(sample_data: xr.DataArray):
    size = {"x": 3, "y": 5}
    result = boxcar_filter_arr(sample_data, size=size, use_pixel=True)
    assert isinstance(result, xr.DataArray)
    assert result.shape == sample_data.shape


def test_boxcar_filter_arr_with_physical_units(sample_data: xr.DataArray):
    size = {"x": 2.0, "y": 4.0}
    result = boxcar_filter_arr(sample_data, size=size, use_pixel=False)
    assert isinstance(result, xr.DataArray)
    assert result.shape == sample_data.shape


def test_boxcar_filter_arr_repeat(sample_data: xr.DataArray):
    size = {"x": 3, "y": 5}
    result = boxcar_filter_arr(sample_data, size=size, repeat_n=2, use_pixel=True)
    assert isinstance(result, xr.DataArray)
    assert result.shape == sample_data.shape


def test_boxcar_filter_arr():
    # Create a sample DataArray
    data = np.random.rand(10, 10)
    coords = {"x": np.arange(10), "y": np.arange(10)}
    arr = xr.DataArray(data, coords=coords, dims=["x", "y"])

    # Apply the filter
    filtered = boxcar_filter_arr(arr, size={"x": 2, "y": 2}, repeat_n=1)

    # Assert the output shape is the same
    assert filtered.shape == arr.shape

    # Assert the values are smoothed (not equal to original)
    assert not np.array_equal(filtered.values, arr.values)

    # Test with pixel-based size
    filtered_pixel = boxcar_filter_arr(arr, size={"x": 2, "y": 2}, use_pixel=True)
    assert filtered_pixel.shape == arr.shape
    assert not np.array_equal(filtered_pixel.values, arr.values)


def test_boxcar_filter_arr_no_size():
    # Create a sample DataArray
    data = np.random.rand(10, 10)
    coords = {"x": np.arange(10), "y": np.arange(10)}
    arr = xr.DataArray(data, coords=coords, dims=["x", "y"])

    # Call the function without providing size
    filtered = boxcar_filter_arr(arr, size=None, repeat_n=1)

    # Assert the output shape is the same
    assert filtered.shape == arr.shape


def test_boxcar_filter_arr_missing_dim_in_size():
    # Create a sample DataArray
    data = np.random.rand(10, 10)
    coords = {"x": np.arange(10), "y": np.arange(10)}
    arr = xr.DataArray(data, coords=coords, dims=["x", "y"])

    # Provide size missing one dimension
    filtered = boxcar_filter_arr(arr, size={"x": 2}, repeat_n=1)

    # Assert the output shape is the same
    assert filtered.shape == arr.shape


def test_boxcar_filter_arr_with_id_attr():
    # Create a sample DataArray with an "id" attribute
    data = np.random.rand(10, 10)
    coords = {"x": np.arange(10), "y": np.arange(10)}
    arr = xr.DataArray(data, coords=coords, dims=["x", "y"], attrs={"id": "test_id"})

    # Apply the filter
    filtered = boxcar_filter_arr(arr, size={"x": 2, "y": 2}, repeat_n=1)

    # Assert provenance context is added (if applicable)
    assert filtered.attrs.get("id") != "test_id"
