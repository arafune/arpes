"""Unittest for helper functions in xarray_extensions."""

import numpy as np
import pytest
import xarray as xr

from arpes.xarray_extensions import _check_equal_spacing


def test_check_equal_spacing_exact():
    coords = xr.DataArray([0, 1, 2, 3, 4])
    spacing = _check_equal_spacing(coords, "x")
    assert spacing == 1


def test_check_equal_spacing_approx():
    coords = xr.DataArray([0, 1.01, 2.02, 3.03, 4.04])
    spacing = _check_equal_spacing(coords, "x", atol=0.02)
    assert np.isclose(spacing, 1.01, atol=0.02)


def test_check_equal_spacing_warns():
    coords = xr.DataArray([0, 1, 2, 3.1, 4.1])
    with pytest.warns(UserWarning, match="Coordinate x is not perfectly equally spaced"):
        spacing = _check_equal_spacing(coords, "x")
    assert spacing == 1
