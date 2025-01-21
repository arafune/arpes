"""Test for utilitiy.bz."""

import numpy as np

from arpes.utilities import bz_spec


def test_bz_points_for_hexagonal_lattice() -> None:
    """Test for bz_points_for_hexagonal_lattice."""
    bz_spec_unit = bz_spec.bz_points_for_hexagonal_lattice()
    bz_spec_unit = np.round(bz_spec_unit, decimals=6)
    # Sort by x-axis values first, then by y-axis values
    bz_spec_unit = bz_spec_unit[np.lexsort((bz_spec_unit[:, 1], bz_spec_unit[:, 0]))]
    desired = np.array(
        [
            [-2 / 3, 0],
            [-1 / 3, -np.sqrt(3) / 3],
            [-1 / 3, np.sqrt(3) / 3],
            [1 / 3, -np.sqrt(3) / 3],
            [1 / 3, np.sqrt(3) / 3],
            [2 / 3, 0],
        ],
    )
    np.testing.assert_array_almost_equal(
        bz_spec_unit,
        desired,
        decimal=5,
    )
