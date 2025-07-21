"""Unit test for analysis.xps."""


import numpy as np
import xarray as xr

from arpes.analysis.xps import approximate_core_levels


def test_approximate_core_levels(xps_map: xr.Dataset) -> None:
    """Test the core level approximation function."""
    xps_spectrum = xps_map.spectrum.sum(["x", "y"], keep_attrs=True)
    approx_levels = approximate_core_levels(xps_spectrum, promenance=5)
    np.testing.assert_almost_equal(approx_levels[0], -34.5066501491)
