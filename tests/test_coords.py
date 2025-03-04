"""Unit test for correction/coords.py."""

import numpy as np
import xarray as xr

from arpes.correction import coords


def test_is_equally_spaced(dataarray_cut: xr.DataArray) -> None:
    """Test for is_equally_spaced."""
    coords_phi = dataarray_cut.coords["phi"].values
    assert coords.is_equally_spaced(coords_phi)


def test_adjust_coords_to_limit_smallside(dataarray_cut: xr.DataArray) -> None:
    expand_doords = coords.adjust_coords_to_limit(dataarray_cut, {"phi": 0.21})
    np.testing.assert_array_almost_equal(
        expand_doords["phi"],
        np.array([0.21, 0.21174533, 0.21349066, 0.21523599, 0.21698132, 0.21872665, 0.22047198]),
    )


def test_adjust_coords_to_limit_largeside(dataarray_cut: xr.DataArray) -> None:
    expand_doords = coords.adjust_coords_to_limit(dataarray_cut, {"phi": 0.65})
    np.testing.assert_array_almost_equal(
        expand_doords["phi"],
        np.array([0.64053584, 0.64228117, 0.6440265, 0.64577183, 0.64751716, 0.64926249, 0.651008]),
    )


def test_adjust_coords_inside_range(dataarray_cut: xr.DataArray) -> None:
    expand_doords = coords.adjust_coords_to_limit(dataarray_cut, {"eV": 0.0})
    np.testing.assert_array_almost_equal(expand_doords["eV"], np.array([]))


def test_adjust_coords_to_limit_2D(dataarray_cut: xr.DataArray) -> None:
    expand_doords = coords.adjust_coords_to_limit(dataarray_cut, {"phi": 0.65, "eV": 0.14})
    np.testing.assert_array_almost_equal(
        expand_doords["phi"],
        np.array([0.64053584, 0.64228117, 0.6440265, 0.64577183, 0.64751716, 0.64926249, 0.651008]),
    )
    np.testing.assert_array_almost_equal(
        expand_doords["eV"],
        np.array(
            [0.13255804, 0.13488362, 0.1372092, 0.13953478, 0.14186036],
        ),
    )


def test_stretch_coords(dataarray_cut: xr.DataArray) -> None:
    expand_doords = coords.adjust_coords_to_limit(dataarray_cut, {"phi": 0.65, "eV": 0.14})
    stretched_data = coords.stretch_coords(dataarray_cut, expand_doords)
    assert stretched_data.shape == (247, 245)
    np.testing.assert_array_almost_equal(
        stretched_data.values[0][-5:],
        np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
    )
    assert np.all(np.isnan(stretched_data.values[-5:]))
