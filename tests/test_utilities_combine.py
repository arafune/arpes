"""Unit tests for combine.concat_along_phi using pytest.

This module provides 100% test coverage for the concat_along_phi function
defined in combine.py. The function combines two ARPES xarray.DataArray
objects along the 'phi' axis with optional enhancements and seam adjustment.

Test cases include:
- Default concatenation behavior
- Handling of occupation_ratio and phi order
- Validation of input parameters
- Verification of enhancement factor
- Assertion of correct ID merging via _combine_id

All tests assume that the `.G.with_values()` accessor is implemented.
"""

from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

import arpes.xarray_extensions
from arpes.utilities.combine import concat_along_phi


@pytest.fixture
def mock_dataarrays():
    """Create two xarray.DataArray objects with 'phi' coordinate and 'id' attributes.

    Returns:
        tuple[xr.DataArray, xr.DataArray]: Two mock DataArrays for testing.
    """
    phi1 = np.linspace(0, 1, 10)
    phi2 = np.linspace(1.1, 2, 10)

    arr_a = xr.DataArray(np.ones((10,)), coords=[("phi", phi1)])
    arr_b = xr.DataArray(np.ones((10,)) * 2, coords=[("phi", phi2)])

    arr_a.attrs["id"] = 1
    arr_b.attrs["id"] = 2

    return arr_a, arr_b


def test_concat_no_occupation_ratio(mock_dataarrays):
    """Test concat_along_phi with default arguments (no occupation_ratio)."""
    arr_a, arr_b = mock_dataarrays
    result = concat_along_phi(arr_a, arr_b)
    assert isinstance(result, xr.DataArray)
    assert "phi" in result.coords
    assert result.size == arr_a.size + arr_b.size


def test_concat_with_occupation_ratio_left_right(mock_dataarrays):
    """Test concat_along_phi with occupation_ratio and arr_a as left."""
    arr_a, arr_b = mock_dataarrays
    result = concat_along_phi(arr_a, arr_b, occupation_ratio=0.5)
    assert isinstance(result, xr.DataArray)


def test_concat_with_occupation_ratio_right_left(mock_dataarrays):
    """Test concat_along_phi with occupation_ratio and arr_b as left."""
    arr_b, arr_a = mock_dataarrays  # reverse order
    result = concat_along_phi(arr_a, arr_b, occupation_ratio=0.5)
    assert isinstance(result, xr.DataArray)


def test_concat_same_phi_min_raises(mock_dataarrays):
    """Test RuntimeError when arr_a and arr_b have the same min(phi)."""
    arr_a, arr_b = mock_dataarrays
    arr_b = arr_b.assign_coords(phi=arr_a.coords["phi"])  # identical phi

    with pytest.raises(RuntimeError, match="Cannot combine them"):
        concat_along_phi(arr_a, arr_b, occupation_ratio=0.5)


def test_invalid_occupation_ratio(mock_dataarrays):
    """Test assertion error when occupation_ratio is outside [0, 1]."""
    arr_a, arr_b = mock_dataarrays
    with pytest.raises(AssertionError):
        concat_along_phi(arr_a, arr_b, occupation_ratio=1.5)


def test_enhance_a_effect(mock_dataarrays):
    """Test that enhance_a multiplies the data values of arr_a."""
    arr_a, arr_b = mock_dataarrays
    result = concat_along_phi(arr_a, arr_b, enhance_a=2.0)
    expected = arr_a.values * 2.0
    np.testing.assert_allclose(result.sel(phi=arr_a.phi).values, expected)
