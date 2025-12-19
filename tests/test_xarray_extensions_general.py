"""Unit test for xarray_extensions/accessor/genelral.py."""

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

import arpes.xarray_extensions.accessor.general  # noqa: F401


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


def test_round_coordinates_for_da(simple_da: xr.DataArray) -> None:
    """Simple check for thin wrapper of round_coordinates_impl."""
    rounded = simple_da.G.round_coordinates({"x": [2.1], "y": [0.2]})
    assert isinstance(rounded["x"], float)
    assert isinstance(rounded["y"], float)


def test_filter_coord_for_da(simple_da: xr.DataArray) -> None:
    """Simple test for thin wrapper of filter_coord_impl."""

    def sieve(coord, da):
        return coord > 5.0

    filtered = simple_da.G.filter_coord("x", sieve)
    assert isinstance(filtered, xr.DataArray)
    assert np.all(filtered.coords["x"].values > 5.0)


def test_apply_over_for_da(simple_da: xr.DataArray) -> None:
    """Simple test for thin wrapper of apply_over_impl."""

    def fn(da):
        return da.values * 2

    result = simple_da.G.apply_over(
        fn,
        selections={"x": 0.0},
    )

    assert isinstance(result, xr.DataArray)
    assert result is not simple_da
    assert np.all(result.sel(x=0.0).values == simple_da.sel(x=0.0).values * 2)


def test_filter_vars_basic():
    ds = xr.Dataset({"a": ("x", [1, 2]), "b": ("x", [3, 4]), "temp": ("x", [5, 6])})
    filtered = ds.G.filter_vars(lambda k, v: k.startswith("t"))
    assert set(filtered.data_vars) == {"temp"}
    assert set(ds.data_vars) == {"a", "b", "temp"}  # original unchanged


def test_filter_vars_all_excluded():
    ds = xr.Dataset({"a": ("x", [1, 2])})
    filtered = ds.G.filter_vars(lambda k, v: False)
    assert len(filtered.data_vars) == 0


def test_scale_meshgrid_scalar():
    XX, YY = np.meshgrid(np.arange(2), np.arange(3))
    ds = xr.Dataset({"x_coord": (("y", "x"), XX), "y_coord": (("y", "x"), YY)})
    scaled = ds.G.scale_meshgrid(("x_coord", "y_coord"), scale=2.0)
    assert np.all(scaled["x_coord"].values == XX * 2.0)
    assert np.all(scaled["y_coord"].values == YY * 2.0)


def test_scale_meshgrid_1d_array():
    XX, YY = np.meshgrid(np.arange(2), np.arange(3))
    ds = xr.Dataset({"x_coord": (("y", "x"), XX), "y_coord": (("y", "x"), YY)})
    scale = np.array([0.5, 1.5])
    scaled = ds.G.scale_meshgrid(("x_coord", "y_coord"), scale=scale)
    assert np.all(scaled["x_coord"].values == XX * 0.5)
    assert np.all(scaled["y_coord"].values == YY * 1.5)


def test_scale_meshgrid_matrix():
    XX, YY = np.meshgrid(np.arange(2), np.arange(3))
    ds = xr.Dataset({"x_coord": (("y", "x"), XX), "y_coord": (("y", "x"), YY)})
    # swap x and y
    scale = np.array([[0, 1], [1, 0]])
    scaled = ds.G.scale_meshgrid(("x_coord", "y_coord"), scale=scale)
    assert np.all(scaled["x_coord"].values == YY)
    assert np.all(scaled["y_coord"].values == XX)


def test_clean_outliers_clip():
    arr = xr.DataArray(np.array([1, 2, 100, 200, 5], dtype=float))

    clipped = arr.G.clean_outliers(clip=20)
    low, high = np.percentile(arr.values, [20, 80])

    assert np.all(clipped.values >= low)
    assert np.all(clipped.values <= high)

    assert not np.all(arr.values == clipped.values)


def test_transform_mean_std():
    arr = xr.DataArray(
        np.random.rand(4, 5), dims=("x", "y"), coords={"x": np.arange(4), "y": np.arange(5)}
    )

    def fn(data, coord):
        stats = np.array([data.mean().item(), data.std().item()])
        return xr.DataArray(stats, dims=("stat",), coords={"stat": ["mean", "std"]})

    out = arr.G.transform("x", fn)
    assert out.dims == ("x", "stat")
    assert "mean" in out.coords["stat"].values
    assert "std" in out.coords["stat"].values
    assert out.shape == (4, 2)


def test_map_axes_single_axis():
    arr = xr.DataArray(
        np.arange(6).reshape(2, 3),
        dims=("x", "y"),
        coords={"x": [0, 1], "y": [10, 20, 30]},
    )

    def fn(data, coord):  # data: DataArray, coord: dict
        return data * 2

    mapped = arr.G.map_axes(["x"], fn)
    assert np.all(mapped.values == arr.values * 2)


def test_map_axes_dtype():
    arr = xr.DataArray(
        [1, 2, 3],
        dims=["x"],
        coords={"x": [0, 1, 2]},
    )
    mapped = arr.G.map_axes(
        "x",
        lambda data, coord: data * 2,
        dtype=float,
    )
    assert mapped.dtype == float


def test_shift_coords_by_basic():
    arr = xr.DataArray(
        np.arange(6).reshape(2, 3),
        dims=("x", "y"),
        coords={
            "x": [0, 1],
            "y": [10, 20, 30],
        },
    )
    shifted = arr.G.shift_coords_by({"x": 5, "y": -10})
    assert np.all(shifted.coords["x"].values == arr.coords["x"].values + 5)
    assert np.all(shifted.coords["y"].values == arr.coords["y"].values - 10)


def test_apply_over_dataset_ndarray_return_raises(simple_ds: xr.Dataset) -> None:
    """Dataset + ndarray return should raise TypeError."""

    def fn(ds) -> NDArray[np.float64]:
        return ds["a"].values + 10

    with pytest.raises(TypeError, match="ndarray return is not supported"):
        simple_ds.G.apply_over(
            fn,
            selections={"x": 4.0},
        )


def test_round_coordinates_for_ds(simple_ds: xr.Dataset) -> None:
    """Simple check for thin wrapper of round_coordinates_impl."""
    rounded = simple_ds.G.round_coordinates({"x": [2.1], "y": [0.2]})
    assert isinstance(rounded["x"], float)
    assert isinstance(rounded["y"], float)


def test_filter_coord_for_ds(simple_ds: xr.Dataset) -> None:
    """Simple test for thin wrapper of filter_coord_impl."""

    def sieve(coord, da):
        return coord > 5.0

    filtered = simple_ds.G.filter_coord("x", sieve)
    assert isinstance(filtered, xr.Dataset)
    assert np.all(filtered.coords["x"].values > 5.0)
