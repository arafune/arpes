"""Unit test for correction/coords.py."""

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from arpes.correction import coords


@pytest.mark.parametrize(
    ("limits", "expected"),
    [
        (
            {"phi": 0.21},
            np.array(
                [0.21, 0.21174533, 0.21349066, 0.21523599, 0.21698132, 0.21872665, 0.22047198],
            ),
        ),
        (
            {"phi": 0.65},
            np.array(
                [0.64053584, 0.64228117, 0.6440265, 0.64577183, 0.64751716, 0.64926249, 0.651008],
            ),
        ),
        ({"eV": 0.0}, np.array([])),
        (
            {"phi": 0.65, "eV": 0.14},
            {
                "phi": np.array(
                    [
                        0.64053584,
                        0.64228117,
                        0.6440265,
                        0.64577183,
                        0.64751716,
                        0.64926249,
                        0.651008,
                    ],
                ),
                "eV": np.array([0.13255804, 0.13488362, 0.1372092, 0.13953478, 0.14186036]),
            },
        ),
    ],
)
def test_adjust_coords_to_limit(
    dataarray_cut: xr.DataArray,
    limits: dict[str, float | dict[str, float]],
    expected: NDArray[np.float64] | dict[str, NDArray[np.float64]],
) -> None:
    expand_coords = coords.adjust_coords_to_limit(dataarray_cut, limits)
    if isinstance(expected, dict):
        for key, value in expected.items():
            np.testing.assert_array_almost_equal(expand_coords[key], value)
    else:
        np.testing.assert_array_almost_equal(expand_coords[next(iter(limits.keys()))], expected)


def test_is_equally_spaced(dataarray_cut: xr.DataArray) -> None:
    """Test for is_equally_spaced."""
    coords_phi = dataarray_cut.coords["phi"].values
    assert coords.is_equally_spaced(coords_phi)


def test_is_equally_spaced_exact():
    coords_ = xr.DataArray([0, 1, 2, 3, 4])
    spacing = coords.is_equally_spaced(coords_, "x")
    assert spacing == 1


def test_is_equal_spaced_approx():
    coords_ = xr.DataArray([0, 1.01, 2.02, 3.03, 4.04])
    spacing = coords.is_equally_spaced(coords_, "x", atol=0.02)
    assert np.isclose(spacing, 1.01, atol=0.02)


def test_is_equal_spaced_spacing_warns():
    coords_ = xr.DataArray([0, 1, 2, 3.1, 4.1])
    with pytest.warns(UserWarning, match="Coordinate x is not perfectly equally spaced"):
        spacing = coords.is_equally_spaced(coords_, "x")
    assert spacing == 1


def test_extend_coords(dataarray_cut: xr.DataArray) -> None:
    expand_doords = coords.adjust_coords_to_limit(dataarray_cut, {"phi": 0.65, "eV": 0.14})
    stretched_data = coords.extend_coords(dataarray_cut, expand_doords)
    assert stretched_data.shape == (247, 245)
    np.testing.assert_array_almost_equal(
        stretched_data.values[0][-5:],
        np.array([0, 0, 0, 0, 0]),
    )
    assert np.all(stretched_data.values[-5:] == 0)
