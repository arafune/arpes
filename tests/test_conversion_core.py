import numpy as np
import pytest
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from arpes.utilities.conversion.core import (
    convert_to_kspace,
    grid_interpolator_from_dataarray,
    slice_along_path,
)

# FILE: tests/test_core.py


@pytest.fixture
def sample_dataarray() -> xr.DataArray:
    rng = np.random.default_rng()
    return xr.DataArray(
        rng.random((10, 10)),
        dims=["x", "y"],
        coords={"x": np.linspace(0, 9, 10), "y": np.linspace(0, 9, 10)},
    )


@pytest.mark.skip
def test_grid_interpolator_from_dataarray_linear(sample_dataarray: xr.DataArray) -> None:
    interpolator = grid_interpolator_from_dataarray(sample_dataarray, method="linear")
    assert isinstance(interpolator, RegularGridInterpolator)


def test_grid_interpolator_from_dataarray_nearest(sample_dataarray: xr.DataArray) -> None:
    interpolator = grid_interpolator_from_dataarray(sample_dataarray, method="nearest")
    assert isinstance(interpolator, RegularGridInterpolator)


def test_grid_interpolator_from_dataarray_invalid_method(sample_dataarray: xr.DataArray) -> None:
    with pytest.raises(ValueError):
        grid_interpolator_from_dataarray(sample_dataarray, method="invalid")


@pytest.mark.skip
def test_slice_along_path(sample_dataarray: xr.DataArray) -> None:
    path = [{"x": 0, "y": 0}, {"x": 9, "y": 9}]
    result = slice_along_path(sample_dataarray, path)
    assert isinstance(result, xr.Dataset)


def test_convert_to_kspace(sample_dataarray: xr.DataArray) -> None:
    result = convert_to_kspace(sample_dataarray)
    assert isinstance(result, xr.DataArray)
