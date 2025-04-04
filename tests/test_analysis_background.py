"""Unit test for analysis.background.py."""

import numpy as np
import pytest
import xarray as xr
from lmfit.lineshapes import lorentzian
from numpy.typing import NDArray

from arpes.analysis.background import calculate_background_hull, remove_background_hull


@pytest.fixture
def lorentzian_curve() -> NDArray[np.float64]:
    x = np.linspace(-2, 0, 100)
    return lorentzian(x, 0.05, -0.5, 0.1)


@pytest.fixture
def lorentzian_linear_bg(lorentzian_curve: NDArray[np.float64]):
    x = np.linspace(-2, 0, 100)
    y = x + lorentzian_curve
    return xr.DataArray(y, coords={"eV": x}, dims="eV")


def test_background_hull_linear(lorentzian_linear_bg: xr.DataArray):
    bg = calculate_background_hull(lorentzian_linear_bg)
    x = np.linspace(-2, 0, 100)
    lor = lorentzian(x, 0.05, -0.5, 0.1)
    np.testing.assert_allclose(
        bg.values,
        lorentzian_linear_bg.values - lor,
        atol=1e-2,
        rtol=1e-2,
    )


def test_remove_background_hull_linear(lorentzian_linear_bg: xr.DataArray):
    removed = remove_background_hull(lorentzian_linear_bg)
    x = np.linspace(-2, 0, 100)
    lor = lorentzian(x, 0.05, -0.5, 0.1)

    np.testing.assert_allclose(
        removed,
        lor,
        atol=1e-2,
        rtol=1e-2,
    )


def test_invalid_input():
    with pytest.raises(AssertionError):
        calculate_background_hull(np.array([1, 2, 3]))
