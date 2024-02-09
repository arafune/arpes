"""Test for utilitiy.bz."""

import numpy as np

from arpes.utilities import bz


def test_as_2d() -> None:
    """Test for ''bz.as_2d'."""
    points_3d = np.array([[0, 1, 2], [2, 3, 4]])
    np.testing.assert_array_equal(bz.as_2d(points_3d), np.array([[0, 1], [2, 3]]))


def test_orthorombic_cell() -> None:
    """Test for orthormbic_cell function."""
    np.testing.assert_array_equal(
        bz.orthorhombic_cell(5.4, 2.3, 3.0),
        np.array(
            [[5.4, 0, 0], [0, 2.3, 0], [0, 0, 3.0]],
        ),
    )


def test_hex_cell() -> None:
    """Test for hex cell function."""
    np.testing.assert_array_equal(
        bz.hex_cell(3.0, 4.0),
        np.array(
            [
                [3.0, 0.0, 0.0],
                [-1.5, 2.598076211353316, 0],
                [0, 0, 4.0],
            ],
        ),
    )

    np.testing.assert_array_equal(
        bz.hex_cell_2d(3.0),
        np.array(
            [
                [3.0, 0.0],
                [-1.5, 2.598076211353316],
            ],
        ),
    )


def test_special_point_to_vector() -> None:
    """Test for special_point_to_vector.

    FCC bulk lattice is used for test.
    """
    icell = np.array([[1.0, -1.0, 1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, 1.0]])
    special_point_k = bz.SpecialPoint("K", negate=False, bz_coord=(0.375, 0.375, 0.75))

    #    """Test for parse_single_path."""
    # def test_single_path() -> None:
    #    path = "GKM"
    #    assert bz.parse_single_path(path) == [[0, 0, 0], [1, 2, 3], [3, 4, 5]]
