"""Test for utilitiy.bz."""

import numpy as np

from arpes.utilities import bz_spec


def test_bz_points_for_hexagonal_lattice() -> None:
    """Test for bz_points_for_hexagonal_lattice."""
    np.testing.assert_array_almost_equal(
        bz_spec.bz_points_for_hexagonal_lattice(),
        np.array(
            [
                [1 / 3, -np.sqrt(3) / 3],
                [-1 / 3, -np.sqrt(3) / 3],
                [-2 / 3, 0],
                [-1 / 3, np.sqrt(3) / 3],
                [1 / 3, np.sqrt(3) / 3],
                [2 / 3, 0],
            ],
        ),
    )
