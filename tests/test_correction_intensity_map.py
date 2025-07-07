import numpy as np
import pytest
import xarray as xr

from arpes.correction.intensity_map import shift


@pytest.fixture
def sample_data():
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 5, 6)
    z = np.random.rand(len(y), len(x))
    return xr.DataArray(z, coords={"y": y, "x": x}, dims=["y", "x"])


@pytest.fixture
def sample_data3D():
    x = np.linspace(0, 10, 11)
    y = np.linspace(0, 5, 6)
    w = np.linspace(0, 1, 6)  # Adding a third dimension for testing
    z = np.random.rand(len(y), len(x), len(w))
    return xr.DataArray(z, coords={"y": y, "x": x, "w": w}, dims=["y", "x", "w"])


def test_shift_with_xrdataarray(sample_data):
    shift_vals = xr.DataArray(
        np.ones(sample_data.sizes["y"]), coords={"y": sample_data.coords["y"]}, dims=["y"],
    )
    out = shift(sample_data, shift_vals, shift_axis="x", shift_coords=False)
    assert isinstance(out, xr.DataArray)
    assert out.shape == sample_data.shape
    np.testing.assert_array_equal(out.coords["y"], sample_data.coords["y"])


def test_shift_with_xrdataarray_shift_coords(sample_data):
    shift_vals = xr.DataArray(
        np.ones(sample_data.sizes["y"]), coords={"y": sample_data.coords["y"]}, dims=["y"],
    )
    out = shift(sample_data, shift_vals, shift_axis="x", shift_coords=True)
    assert not np.allclose(out.coords["x"], sample_data.coords["x"])


def test_shift_with_ndarray(sample_data):
    shift_vals = np.ones(sample_data.sizes["y"])
    out = shift(sample_data, shift_vals, shift_axis="x", by_axis="y")
    assert out.shape == sample_data.shape


def test_shift_with_ndarray_missing_by_axis(sample_data):
    shift_vals = np.ones(sample_data.sizes["y"])
    # This should succeed because the function infers by_axis when 2D
    out = shift(sample_data, shift_vals, shift_axis="x")
    assert out.shape == sample_data.shape


def test_shift_coords_alignment(sample_data):
    shift_vals = np.linspace(-1, 1, sample_data.sizes["y"])
    out = shift(sample_data, shift_vals, shift_axis="x", by_axis="y", shift_coords=True)
    mean_shift = np.mean(shift_vals)
    expected_coords = sample_data.coords["x"] + mean_shift
    np.testing.assert_allclose(out.coords["x"], expected_coords, atol=1e-6)


def test_shift_extend_coords_min(sample_data):
    shift_vals = np.full(sample_data.sizes["y"], 5.0)
    out = shift(sample_data, shift_vals, shift_axis="x", by_axis="y", extend_coords=True)
    assert out.sizes["x"] > sample_data.sizes["x"]


def test_shift_extend_coords_max(sample_data):
    shift_vals = np.full(sample_data.sizes["y"], -5.0)
    out = shift(sample_data, shift_vals, shift_axis="x", by_axis="y", extend_coords=True)
    assert out.sizes["x"] > sample_data.sizes["x"]


def test_shift_axis_required(sample_data):
    shift_vals = np.ones(sample_data.sizes["y"])
    with pytest.raises(AssertionError):
        shift(sample_data, shift_vals, shift_axis="")


def test_shift_by_axis_required_for_ndarray(sample_data3D):
    shift_vals = np.ones(sample_data3D.sizes["x"])  # Not matching y
    with pytest.raises(TypeError):
        shift(sample_data3D, shift_vals, shift_axis="y")


def test_shift_with_integer_array():
    x = np.arange(5)
    y = np.arange(4)
    z = np.ones((len(y), len(x)), dtype=int)
    data = xr.DataArray(z, coords={"y": y, "x": x}, dims=["y", "x"])
    shift_vals = np.array([1.0] * len(y))
    out = shift(data, shift_vals, shift_axis="x", by_axis="y")
    assert np.issubdtype(out.dtype, np.integer)


def test_nan_padding_with_float_array():
    x = np.arange(5)
    y = np.arange(4)
    z = np.ones((len(y), len(x)), dtype=float)
    data = xr.DataArray(z, coords={"y": y, "x": x}, dims=["y", "x"])
    shift_vals = np.array([1.0] * len(y))
    out = shift(data, shift_vals, shift_axis="x", by_axis="y")
    assert np.isnan(out.values).any()
