"""Unit test for axis_preparation.py."""

import numpy as np
import pytest
import xarray as xr

import arpes.xarray_extensions  # noqa: F401
from arpes.preparation.axis_preparation import (
    dim_normalizer,
    flip_axis,
    normalize_dim,
    normalize_max,
    normalize_total,
    sort_axis,
    transform_dataarray_axis,
    vstack_data,
)

# --- Fixtures ---


@pytest.fixture
def sample_array():
    coords = {"x": np.linspace(0, 4, 5), "y": np.linspace(0, 2, 3)}
    data = np.random.rand(3, 5)
    return xr.DataArray(data, coords=coords, dims=["y", "x"], attrs={"id": "sample"})


@pytest.fixture
def sample_dataset(sample_array: xr.DataArray):
    return xr.Dataset({"intensity": sample_array})


# --- Tests ---


def test_sort_axis(sample_array: xr.DataArray):
    reversed_data = sample_array.sel(x=slice(None, None, -1))
    sorted_data = sort_axis(reversed_data, "x")
    assert np.all(sorted_data.x.values == np.sort(sample_array.x.values))
    assert sorted_data.shape == sample_array.shape


def test_flip_axis(sample_array: xr.DataArray):
    flipped = flip_axis(sample_array, "x")
    assert np.allclose(flipped.x.values, sample_array.x.values[::-1])
    assert np.allclose(flipped.values, np.flip(sample_array.values, axis=1))


def test_flip_axis_only_coords(sample_array: xr.DataArray):
    flipped = flip_axis(sample_array, "x", flip_data=False)
    assert np.allclose(flipped.values, sample_array.values)
    assert np.allclose(flipped.x.values, sample_array.x.values[::-1])


@pytest.mark.skip
def test_normalize_dim(sample_array: xr.DataArray):
    normed = normalize_dim(sample_array, "x")
    avg = normed.sum(dim="y").mean().item()
    assert np.isclose(avg, 1, rtol=1e-5)


@pytest.mark.skip
def test_dim_normalizer_function(sample_array: xr.DataArray):
    norm_fn = dim_normalizer("x")
    result = norm_fn(sample_array)
    assert "x" in result.dims
    assert np.isclose(result.sum(["y"]).mean().item(), 1, rtol=1e-5)


def test_normalize_total(sample_array: xr.DataArray):
    total = 1_000_000
    result = normalize_total(sample_array, total_intensity=total)
    assert np.isclose(result.sum(), total, rtol=1e-4)


def test_vstack_data_attrs():
    arrs = [
        xr.DataArray(
            np.full((2, 2), fill_value=i),
            coords={"x": [0, 1], "y": [0, 1]},
            dims=["y", "x"],
            attrs={"z": i},
        )
        for i in range(3)
    ]
    result = vstack_data(arrs, "z")
    assert "z" in result.dims
    assert result.shape[0] == 3
    assert "z" not in result.attrs


def test_vstack_data_coords():
    arrs = [
        xr.DataArray(
            np.full((2, 2), fill_value=i),
            coords={"x": [0, 1], "y": [0, 1], "z": i},
            dims=["y", "x"],
        )
        for i in range(3)
    ]
    result = vstack_data(arrs, "z")
    assert "z" in result.dims
    assert result.shape[0] == 3


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


@pytest.mark.skip
def test_transform_dataarray_axis(sample_dataset: xr.DataArray):
    def identity_transform(arr: xr.DataArray, axis: int):
        # return original coordinate index values (for testing)
        return np.indices(arr.shape)[axis]

    new_axis = np.linspace(-1, 1, sample_dataset.dims["x"])
    new_ds = transform_dataarray_axis(
        func=identity_transform,
        old_and_new_axis_names=("x", "kx"),
        new_axis=new_axis,
        dataset=sample_dataset,
        prep_name=lambda name: f"{name}_kx",
        remove_old=True,
    )

    assert "kx" in new_ds.coords
    assert "intensity_kx" in new_ds.data_vars
    assert "x" not in new_ds.dims


def test_normalize_max_default(dataarray_cut: xr.DataArray):
    """absolute=False, keep_attrs=True, max_value=1.0."""
    result = normalize_max(dataarray_cut)
    expected = 1.0
    assert result.values.max() == expected


def test_normalize_max_with_absolute(dataarray_cut: xr.DataArray):
    """absolute=True."""
    dataarray_cut = -dataarray_cut
    result = normalize_max(dataarray_cut, absolute=True)
    expected = -1.0
    assert result.values.min() == expected


def test_normalize_max_with_max_value(dataarray_cut: xr.DataArray):
    """max_value=2.5."""
    result = normalize_max(dataarray_cut, max_value=2.5)
    expected = 2.5
    assert result.values.max() == expected


def test_normalize_max_without_attrs(dataarray_cut: xr.DataArray):
    """keep_attrs=False."""
    result = normalize_max(dataarray_cut, keep_attrs=False)
    assert len(result.attrs) == 2


def test_normalize_max_return_type(dataarray_cut: xr.DataArray):
    """xr.DataArray."""
    result = normalize_max(dataarray_cut)
    assert isinstance(result, xr.DataArray)
