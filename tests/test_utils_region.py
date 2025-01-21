"""Unit test for utilities/region.py."""

import numpy as np
import xarray as xr

from arpes.utilities.region import (
    find_spectrum_angular_edges,
    find_spectrum_angular_edges_full,
    find_spectrum_energy_edges,
    meso_effective_selector,
    wide_angle_selector,
)


def test_find_spectrum_energy_edges(dataarray_cut: xr.DataArray) -> None:
    """Test for find_spectrum_energy_edges."""
    np.testing.assert_allclose(
        np.array([-0.3883721, -0.14883726, 0.00465109]),
        find_spectrum_energy_edges(dataarray_cut),
        rtol=1e-5,
    )
    np.testing.assert_array_equal(
        np.array([16, 119, 185]),
        find_spectrum_energy_edges(dataarray_cut, indices=True),
    )


def test_find_spectrum_angular_edges(dataarray_cut: xr.DataArray) -> None:
    """Test for find_spectrum_angular_edges."""
    np.testing.assert_allclose(
        np.array([0.249582, 0.350811, 0.385718, 0.577704]),
        find_spectrum_angular_edges(dataarray_cut),
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        np.array([16, 74, 94, 204]),
        find_spectrum_angular_edges(dataarray_cut, indices=True),
    )


def test_wide_angle_selector(dataarray_cut: xr.DataArray) -> None:
    """Test for wide_angle_selector."""
    assert wide_angle_selector(dataarray_cut) == slice(
        np.float64(0.2995820830351892),
        np.float64(0.5277039824101235),
        None,
    )


def test_meso_effective_selector(dataarray_cut: xr.DataArray) -> None:
    """Test for meso_effective_selector."""
    assert meso_effective_selector(dataarray_cut) == slice(
        np.float64(-0.2953489149999961),
        np.float64(-0.09534891499999612),
        None,
    )


def test_find_spectrum_angular_edges_full(dataarray_cut: xr.DataArray) -> None:
    """Test for find_spectrum_angular_edges_full."""
    desired = (
        np.array([0.26964973, 0.26092309, 0.30281099, 0.28012171, 0.26441375, 0.25394177]),
        np.array([0.59951696, 0.60824361, 0.60998894, 0.60824361, 0.54890241, 0.56810104]),
    )
    results = find_spectrum_angular_edges_full(dataarray_cut)
    np.testing.assert_allclose(
        results[0],
        desired[0],
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        results[1],
        desired[1],
        rtol=1e-5,
    )
    assert isinstance(results[2], xr.DataArray)
