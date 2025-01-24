"""Test for data loading."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import xarray as xr

from arpes.endstations.plugin.ALG_main import ALGMainChamber
from arpes.io import load_data, load_example_data


def test_load_data() -> None:
    """Test direct loading of fits data using ALG-MC.

    This test function loads data from a specified FITS file located in the
    'resources/datasets/basic' directory and checks if the loaded data is an
    instance of an xarray.Dataset. It also verifies that the shape of the
    spectrum data is (240, 240).

    Args:
        sandbox_configuration ([TODO:type]): [TODO:description]
    """
    test_data_location = (
        Path(__file__).parent / "resources" / "datasets" / "basic" / "main_chamber_cut_0.fits"
    )

    data = load_data(file=test_data_location, location="ALG-MC")

    assert isinstance(data, xr.Dataset)
    assert data.spectrum.shape == (240, 240)


def test_load_data_with_plugin_specified() -> None:
    """[TODO:summary].

    [TODO:description]

    Args:
        sandbox_configuration ([TODO:type]): [TODO:description]
    """
    test_data_location = (
        Path(__file__).parent / "resources" / "datasets" / "basic" / "main_chamber_cut_0.fits"
    )

    data = load_data(file=test_data_location, location="ALG-MC")
    directly_specified_data = load_data(file=test_data_location, location=ALGMainChamber)

    assert isinstance(directly_specified_data, xr.Dataset)
    assert directly_specified_data.spectrum.shape == (240, 240)
    assert np.all(data.spectrum.values == directly_specified_data.spectrum.values)


@pytest.mark.parametrize(
    ("data_name", "expected_shape"),
    [
        ("cut", (240, 240)),
        ("cut2", (600, 501)),
        ("cut3", (137, 82)),
        ("map", (81, 150, 111)),
        ("map2", (137, 82, 116)),
    ],
    ids=["cut", "cut2", "cut3", "map", "map2"],
    # Description: Parametrize test cases for different example data and their expected shapes
)

def test_load_example_map_data(
    data_name: Literal["cut", "cut2", "cut3", "map", "map2"],
    expected_shape: tuple[int, int] | tuple[int, int, int],
    ) -> None:
    """Test loading example ARPES map data.

    This test validates the loading of example data in a form of:
        - 2D cuts: Two-dimensional data slices.
        - 3D maps: Three-dimensional datasets.

    Parameters:
    data_name (Literal["cut", "cut2", "cut3, "map", "map2"]): The name of the example data to load.
    expected_shape (tuple[int, int] | tuple[int, int, int]): The expected shape of the data.

    Asserts:
    - The loaded data is an xarray Dataset.
    - The 'spectrum' in the dataset is an xarray DataArray.
    - The shape of the 'spectrum' matches the expected shape.
    - All necessary coordinates:
        ('phi', 'psi', 'alpha', 'chi', 'beta', 'theta', 'x', 'y', 'z', 'hv')
        are present in the data.
    """
    data = load_example_data(data_name)

    # check that the data is an xarray dataset
    assert isinstance(data, xr.Dataset)
    assert isinstance(data.spectrum, xr.DataArray)

    # Verify the shape of 'spectrum' matches the expected shape
    assert data.spectrum.shape == expected_shape, (
        f"{data_name}: Shape mismatch. "
        f"Expected {expected_shape}, got {data.spectrum.shape}"
    )

    # Verify all necessary coordinates are present
    necessary_coords = {"phi", "psi", "alpha", "chi", "beta", "theta", "x", "y", "z", "hv"}
    missing_coords = necessary_coords - set(map(str, data.coords))
    assert not missing_coords, f"{data_name}: Missing coordinates: {missing_coords}"
