"""Unit test of io module in aryspes."""

import json
from pathlib import Path

import pytest
import xarray as xr

from arpes.io import load_custom_netcdf, load_example_data, save_custom_netcdf


def test_load_example_raises_kye_error() -> None:
    msg = "Could not find requested example_name: cut0.*"
    with pytest.raises(KeyError, match=msg):
        load_example_data("cut0")


@pytest.fixture
def sample_dataarray() -> xr.DataArray:
    """Fixture to provide a sample xarray.DataArray for testing."""
    data = xr.DataArray([1, 2, 3], dims=["x"], coords={"x": [0, 1, 2]})
    data.attrs = {"description": "Test DataArray", "info": {"nested": "value"}}
    return data


@pytest.fixture
def sample_dataset() -> xr.Dataset:
    """Fixture to provide a sample xarray.Dataset for testing."""
    data = xr.Dataset({"var": ("x", [1, 2, 3])}, coords={"x": [0, 1, 2]})
    data.attrs = {"description": "Test Dataset", "info": {"nested": "value"}}
    return data


@pytest.fixture
def sample_datatree() -> xr.DataTree:
    """Fixture to provide a sample xarray.DataTree for testing."""
    data = xr.DataTree({"node1": xr.DataArray([1, 2]), "node2": xr.DataArray([3, 4])})
    data.attrs = {"description": "Test DataTree", "info": {"nested": "value"}}
    return data


def test_save_load_dataarray(sample_dataarray: xr.DataArray, tmp_path: Path):
    """Test saving and loading a DataArray with JSON-encoded attrs."""
    file_path = tmp_path / "dataarray.nc"

    # Save the DataArray
    save_custom_netcdf(sample_dataarray, file_path)

    # Load the DataArray
    loaded = load_custom_netcdf(file_path)

    assert isinstance(loaded, xr.DataArray)
    assert loaded.attrs["description"] == "Test DataArray"
    assert json.loads(loaded.attrs["info"]) == {"nested": "value"}


def test_save_load_dataset(sample_dataset: xr.Dataset, tmp_path: Path):
    """Test saving and loading a Dataset with JSON-encoded attrs."""
    file_path = tmp_path / "dataset.nc"

    # Save the Dataset
    save_custom_netcdf(sample_dataset, file_path)

    # Load the Dataset
    loaded = load_custom_netcdf(file_path)

    assert isinstance(loaded, xr.Dataset)
    assert loaded.attrs["description"] == "Test Dataset"
    assert json.loads(loaded.attrs["info"]) == {"nested": "value"}


def test_save_load_datatree(sample_datatree: xr.DataTree, tmp_path: Path):
    """Test saving and loading a DataTree with JSON-encoded attrs."""
    file_path = tmp_path / "datatree.nc"

    # Save the DataTree
    save_custom_netcdf(sample_datatree, file_path)

    # Load the DataTree
    loaded = load_custom_netcdf(file_path)

    assert isinstance(loaded, xr.DataTree)
    assert loaded.attrs["description"] == "Test DataTree"
    assert json.loads(loaded.attrs["info"]) == {"nested": "value"}


def test_save_load_with_kwargs(sample_dataarray: xr.DataArray, tmp_path: Path):
    """Test saving and loading with additional kwargs passed to `to_netcdf`."""
    file_path = tmp_path / "dataarray_with_kwargs.nc"

    # Save with engine kwargs
    save_custom_netcdf(sample_dataarray, file_path, engine="h5netcdf")

    # Load the DataArray
    loaded = load_custom_netcdf(file_path)

    assert isinstance(loaded, xr.DataArray)
    assert loaded.attrs["description"] == "Test DataArray"
    assert json.loads(loaded.attrs["info"]) == {"nested": "value"}
