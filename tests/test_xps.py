"""Unit test for xps.py."""

import numpy as np
import xarray as xr

import arpes.xarray_extensions
from arpes.analysis.xps import approximate_core_levels


class TestXPS:
    """Test class for xps analysis.

    While the test data is not XPS, but it's not a problem to test.
    """

    def test_approximate_core_levels(self, dataarray_cut2: xr.DataArray) -> None:
        """Test for approximate_core_levels."""
        energies = approximate_core_levels(dataarray_cut2.S.sum_other(["eV"]))
        np.testing.assert_allclose(
            energies,
            [9.36826347305, 9.58383233535, 9.89520958085],
        )
