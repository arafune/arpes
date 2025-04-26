import pytest
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter
from src.arpes.analysis.filters import boxcar_filter_arr


@pytest.fixture
def sample_data():
    coords = {"x": np.linspace(0, 10, 11), "y": np.linspace(0, 20, 21)}
    data = np.random.random((11, 21))
    return xr.DataArray(data, coords=coords, dims=["x", "y"])


def test_boxcar_filter_arr_with_pixel_units(sample_data):
    size = {"x": 3, "y": 5}
    result = boxcar_filter_arr(sample_data, size=size, use_pixel=True)
    assert isinstance(result, xr.DataArray)
    assert result.shape == sample_data.shape


def test_boxcar_filter_arr_with_physical_units(sample_data):
    size = {"x": 2.0, "y": 4.0}
    result = boxcar_filter_arr(sample_data, size=size, use_pixel=False)
    assert isinstance(result, xr.DataArray)
    assert result.shape == sample_data.shape


def test_boxcar_filter_arr_repeat(sample_data):
    size = {"x": 3, "y": 5}
    result = boxcar_filter_arr(sample_data, size=size, repeat_n=2, use_pixel=True)
    assert isinstance(result, xr.DataArray)
    assert result.shape == sample_data.shape
