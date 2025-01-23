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
    """[TODO:summary].

    [TODO:description]

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


@pytest.mark.parametrize(("data_name", "expected_shape"), [
    ("cut", (240, 240)),
    ("cut2", (600, 501)),
    ("map", (81, 150, 111)),
    ("map2", (137, 82, 116))],
    ids=["cut", "cut2", "map", "map2"])


def test_load_example_data(
    data_name: Literal["cut", "cut2", "map", "map2"],
    expected_shape: tuple[int, int] | tuple[int, int, int],
    ) -> None:
    """Test loading example data for different types."""
    data = load_example_data(data_name)

    # check that the data is an xarray dataset
    assert isinstance(data, xr.Dataset)
    assert isinstance(data.spectrum, xr.DataArray)

    # check that the data has the expected shape
    assert data.spectrum.shape == expected_shape

    # assert that all necessary coordinates are present
    necessary_coords = {"phi", "psi", "alpha", "chi", "beta", "theta", "x", "y", "z", "hv"}
    for necessary_coord in necessary_coords:
        assert necessary_coord in data.coords
